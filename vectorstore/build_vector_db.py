import csv
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "indian_criminal_law"
CSV_FILES = [
    str(BASE_DIR.parent / "data" / "BNS_sections.csv"),
    str(BASE_DIR.parent / "data" / "BNSS_sections.csv"),
    str(BASE_DIR.parent / "data" / "BSA_sections.csv"),
]


def load_rows():
    rows = []
    for fname in CSV_FILES:
        path = Path(fname)
        if not path.exists():
            print(f"WARNING: {fname} not found, skipping.")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def main():
    rows = load_rows()
    if not rows:
        sys.exit("No rows loaded — check that the CSVs are in this folder.")

    print(f"Loaded {len(rows)} sections across {len(CSV_FILES)} files.")

    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # Fresh start each run so re-ingesting doesn't create duplicates.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "BNS, BNSS, BSA statute sections"},
    )

    ids, documents, metadatas = [], [], []
    for row in rows:
        chunk_id = f"{row['act']}_{row['section_number']}"
        ids.append(chunk_id)
        documents.append(row["text"])
        metadatas.append({
            "act": row["act"],
            "section_number": row["section_number"],
            "chapter_number": row["chapter_number"],
            "chapter_title": row["chapter_title"],
            "source_url": row["source_url"],
            "page_number": row["page_number"],
        })

    # Batch to keep memory/requests reasonable.
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
        print(f"  ingested {min(i + batch_size, len(ids))}/{len(ids)}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")
    print(f"DB persisted at: {Path(DB_PATH).resolve()}")


if __name__ == "__main__":
    main()