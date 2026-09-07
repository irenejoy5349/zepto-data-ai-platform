import os
import shutil

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# PATHS
# =========================================================

DOCS_DIR = os.path.join(
    "support_assistant",
    "docs"
)

CHROMA_DIR = os.path.join(
    "support_assistant",
    "chroma_db"
)

COLLECTION_NAME = "zepto_support"


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("=" * 70)
print("SUPPORT ASSISTANT: CHROMA INGESTION")
print("=" * 70)

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Embedding model loaded: all-MiniLM-L6-v2"
)


# =========================================================
# CONNECT TO PERSISTENT CHROMADB
# =========================================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)


# ---------------------------------------------------------
# Fresh collection for reproducible ingestion
# ---------------------------------------------------------

try:
    client.delete_collection(
        name=COLLECTION_NAME
    )
    print(
        f"Existing collection '{COLLECTION_NAME}' deleted."
    )
except Exception:
    pass


collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "Zepto customer support knowledge base"
    }
)


# =========================================================
# READ SUPPORT DOCUMENTS
# =========================================================

documents = []
metadatas = []
ids = []

filenames = sorted(
    filename
    for filename in os.listdir(DOCS_DIR)
    if filename.endswith(".txt")
)


print("\n" + "-" * 70)
print("DOCUMENT INGESTION")
print("-" * 70)


for index, filename in enumerate(filenames):

    file_path = os.path.join(
        DOCS_DIR,
        filename
    )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read().strip()


    # -----------------------------------------------------
    # Keep each support document as one knowledge chunk.
    # Documents are small and topic-focused.
    # -----------------------------------------------------

    document_id = (
        f"doc_{index + 1:03d}"
    )

    documents.append(
        text
    )

    ids.append(
        document_id
    )

    metadatas.append(
        {
            "source": filename,
            "document_id": document_id
        }
    )

    print(
        f"{document_id}: {filename}"
    )


# =========================================================
# GENERATE EMBEDDINGS
# =========================================================

print("\n" + "-" * 70)
print("GENERATING EMBEDDINGS")
print("-" * 70)

embeddings = model.encode(
    documents,
    normalize_embeddings=True
)

print(
    f"Documents embedded: {len(embeddings)}"
)

print(
    f"Embedding dimension: {embeddings.shape[1]}"
)


# =========================================================
# STORE IN CHROMADB
# =========================================================

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


# =========================================================
# VERIFY COLLECTION
# =========================================================

stored_count = collection.count()

print("\n" + "-" * 70)
print("CHROMADB VERIFICATION")
print("-" * 70)

print(
    f"Collection: {COLLECTION_NAME}"
)

print(
    f"Stored documents: {stored_count}"
)


# =========================================================
# SAMPLE RETRIEVAL TEST
# =========================================================

test_query = (
    "How long does a refund take?"
)

query_embedding = model.encode(
    [test_query],
    normalize_embeddings=True
)[0]


results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=3
)


print("\n" + "-" * 70)
print("SAMPLE RETRIEVAL TEST")
print("-" * 70)

print(
    f"Query: {test_query}"
)


for index, (doc_id, metadata) in enumerate(
    zip(
        results["ids"][0],
        results["metadatas"][0]
    ),
    start=1
):

    distance = results["distances"][0][index - 1]

    print(
        f"{index}. {doc_id} | "
        f"{metadata['source']} | "
        f"distance={distance:.4f}"
    )


# =========================================================
# SAVE INGESTION REPORT
# =========================================================

REPORT_PATH = os.path.join(
    "support_assistant",
    "ingestion_report.txt"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ZEpto SUPPORT ASSISTANT - INGESTION REPORT\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "Embedding model: all-MiniLM-L6-v2\n"
    )

    file.write(
        f"Embedding dimension: {embeddings.shape[1]}\n"
    )

    file.write(
        f"Documents ingested: {len(documents)}\n"
    )

    file.write(
        f"Documents stored in ChromaDB: {stored_count}\n"
    )

    file.write(
        f"Collection: {COLLECTION_NAME}\n"
    )

    file.write(
        f"Persistent database path: {CHROMA_DIR}\n\n"
    )

    file.write(
        "Documents\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    for document_id, filename in zip(
        ids,
        filenames
    ):

        file.write(
            f"{document_id}: {filename}\n"
        )

    file.write("\n")

    file.write(
        "Sample retrieval query\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        f"{test_query}\n\n"
    )

    for index, (doc_id, metadata) in enumerate(
        zip(
            results["ids"][0],
            results["metadatas"][0]
        ),
        start=1
    ):

        distance = results["distances"][0][index - 1]

        file.write(
            f"{index}. {doc_id} - "
            f"{metadata['source']} - "
            f"distance={distance:.6f}\n"
        )


print("\nIngestion report saved to:")
print(REPORT_PATH)


print("\n" + "=" * 70)
print("CHROMA INGESTION COMPLETE")
print("=" * 70)