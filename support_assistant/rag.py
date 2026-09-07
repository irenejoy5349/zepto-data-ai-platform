import os
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# PATHS
# =========================================================

CHROMA_DIR = os.path.join(
    "support_assistant",
    "chroma_db"
)

COLLECTION_NAME = "zepto_support"


# =========================================================
# RETRIEVAL COMPONENTS
# =========================================================

print("=" * 70)
print("SUPPORT ASSISTANT: RAG RETRIEVAL")
print("=" * 70)

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Embedding model loaded successfully."
)


# Persistent ChromaDB
client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Chroma collection loaded: {COLLECTION_NAME}"
)

print(
    f"Documents available: {collection.count()}"
)


# =========================================================
# PROMPT SKELETON
# =========================================================

PROMPT_SKELETON = """
ROLE:
You are a helpful Zepto customer support assistant.

CONTEXT:
Use only the information provided in the retrieved support
documents to answer the customer's question.

TASK:
Answer the customer's question using the retrieved context.
If the context does not contain enough information, clearly say
that the available support documents do not provide enough
information.

FORMAT:
Give a direct, concise answer.
Mention relevant source document IDs when appropriate.

LENGTH:
Keep the answer within approximately 80 words unless additional
detail is necessary.

NEGATIVE CONSTRAINT:
Do not invent policies, fees, deadlines, guarantees, or procedures
that are not supported by the retrieved context.

FEW-SHOT EXAMPLE:
User: How long does a refund take?

Retrieved context:
Refunds for eligible orders are normally initiated within
10 working days after approval.

Answer:
Eligible refunds are normally initiated within 10 working days
after approval.
"""


# =========================================================
# RETRIEVE TOP-K DOCUMENTS
# =========================================================

def retrieve_documents(
    query: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant support documents from ChromaDB.
    """

    if not isinstance(query, str):
        raise TypeError(
            "Query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    # Query embedding
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Chroma similarity search
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    result_ids = results.get(
        "ids",
        [[]]
    )[0]

    result_documents = results.get(
        "documents",
        [[]]
    )[0]

    result_metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    result_distances = results.get(
        "distances",
        [[]]
    )[0]


    for index, document_id in enumerate(
        result_ids
    ):

        distance = None

        if index < len(result_distances):
            distance = float(
                result_distances[index]
            )

        metadata = {}

        if index < len(result_metadatas):
            metadata = (
                result_metadatas[index]
                or {}
            )

        document_text = ""

        if index < len(result_documents):
            document_text = (
                result_documents[index]
                or ""
            )

        retrieved.append(
            {
                "document_id": document_id,
                "source": metadata.get(
                    "source",
                    "unknown"
                ),
                "text": document_text,
                "distance": distance
            }
        )

    return retrieved


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(
    retrieved_documents: List[Dict[str, Any]]
) -> str:
    """
    Convert retrieved documents into a structured context block.
    """

    context_parts = []

    for item in retrieved_documents:

        context_parts.append(
            f"[{item['document_id']} | "
            f"{item['source']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# BUILD PROMPT
# =========================================================

def build_prompt(
    query: str,
    retrieved_documents: List[Dict[str, Any]]
) -> str:
    """
    Build the final RAG prompt using the required prompt structure.
    """

    context = build_context(
        retrieved_documents
    )

    return f"""
{PROMPT_SKELETON}

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTION:
Answer using the retrieved context only.
""".strip()


# =========================================================
# SIMPLE RETRIEVAL TEST
# =========================================================

if __name__ == "__main__":

    test_query = (
        "How long does a refund take?"
    )

    print("\n" + "-" * 70)
    print("TEST QUERY")
    print("-" * 70)

    print(
        test_query
    )


    retrieved = retrieve_documents(
        test_query,
        top_k=3
    )


    print("\n" + "-" * 70)
    print("TOP 3 RETRIEVED DOCUMENTS")
    print("-" * 70)


    for rank, item in enumerate(
        retrieved,
        start=1
    ):

        print(
            f"\n{rank}. "
            f"{item['document_id']} | "
            f"{item['source']} | "
            f"distance={item['distance']:.4f}"
        )


    print("\n" + "-" * 70)
    print("GENERATED RAG PROMPT")
    print("-" * 70)

    prompt = build_prompt(
        test_query,
        retrieved
    )

    print(prompt)


    # -----------------------------------------------------
    # Save retrieval test report
    # -----------------------------------------------------

    REPORT_PATH = os.path.join(
        "support_assistant",
        "retrieval_test_report.txt"
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SUPPORT ASSISTANT - RAG RETRIEVAL TEST\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Query: {test_query}\n\n"
        )

        file.write(
            "Top 3 retrieved documents\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for rank, item in enumerate(
            retrieved,
            start=1
        ):

            file.write(
                f"{rank}. "
                f"{item['document_id']} | "
                f"{item['source']} | "
                f"distance={item['distance']:.6f}\n"
            )

        file.write("\n\n")

        file.write(
            "Prompt skeleton\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            prompt
        )


    print(
        f"\nRetrieval test report saved to: "
        f"{REPORT_PATH}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG RETRIEVAL TEST COMPLETE"
    )

    print(
        "=" * 70
    )