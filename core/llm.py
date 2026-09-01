"""
core/llm.py

Groq structured-output helpers. Identical logic in agent.py and
research-pipeline.py (app.py has its own, slightly different,
version left in ui/app.py since it genuinely differs).
"""

import json
import time

from core.config import groq_client


def strict_schema(model_class):
    """
    Convert Pydantic JSON schema into a schema suitable
    for Groq strict structured output.

    Groq requires object schemas to explicitly specify:

        required
        additionalProperties = false

    This function recursively applies those rules.
    """

    schema = model_class.model_json_schema()

    def fix_node(node):
        if isinstance(node, dict):
            # Object node
            if node.get("type") == "object":
                properties = node.get("properties", {})
                node["required"] = list(properties.keys())
                node["additionalProperties"] = False

            # Recursively process:
            # properties, items, $defs, anyOf, etc.
            for value in node.values():
                fix_node(value)
        elif isinstance(node, list):
            for item in node:
                fix_node(item)

    fix_node(schema)
    return schema



def groq_structured(
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema_class,
    max_retries: int = 2,
):
    """
    Call Groq with strict structured output.

    Pydantic validates the final JSON.

    A small retry mechanism is included because occasionally
    a reasoning model may produce an invalid structured response.
    """

    schema = strict_schema(schema_class)
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_class.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
                reasoning_effort="medium",
            )

            content = response.choices[0].message.content
            if not content:
                raise RuntimeError(
                    "Groq returned an empty structured response."
                )

            parsed = json.loads(content)
            validated = schema_class.model_validate(parsed)
            return validated

        except Exception as exc:
            last_error = exc
            err_str = str(exc)

            print()
            print(
                f"[Groq structured call failed "
                f"attempt {attempt + 1}/{max_retries + 1}]"
            )
            print(err_str)

            # On last attempt, if strict schema validation failed,
            # try non-strict JSON repair
            if attempt == max_retries and (
                "json_validate_failed" in err_str
                or "does not match the expected schema" in err_str
            ):
                try:
                    print()
                    print(
                        "[FALLBACK] Strict schema failed. "
                        "Trying non-strict JSON repair..."
                    )

                    response = groq_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},  # Not strict
                        reasoning_effort="medium",
                    )

                    content = response.choices[0].message.content
                    if not content:
                        raise RuntimeError(
                            "Groq returned empty non-strict response."
                        )

                    parsed = json.loads(content)

                    # Repair known omission: missing application array
                    if "legal_research" in parsed:
                        lr = parsed["legal_research"]
                        if "application" not in lr:
                            lr["application"] = []
                            parsed["legal_research"] = lr

                    validated = schema_class.model_validate(parsed)
                    print("[FALLBACK] Repair successful.")
                    return validated

                except Exception as repair_exc:
                    last_error = repair_exc
                    print(f"[FALLBACK FAILED] {repair_exc}")

            if attempt < max_retries:
                time.sleep(1.5)

    raise RuntimeError(
        f"Groq structured call failed after "
        f"{max_retries + 1} attempts: {last_error}"
    )

