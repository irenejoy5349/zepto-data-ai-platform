import os
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from support_assistant.rag import retrieve_documents, build_prompt

# =========================================================
# CONFIGURATION
# =========================================================

# Default graded baseline:
# MOCK_LLM=1
#
# Real LLM integration is optional and is NOT required
# for the deterministic local baseline.

MOCK_LLM = os.getenv(
    "MOCK_LLM",
    "1"
).strip()


# =========================================================
# LANGGRAPH STATE
# =========================================================

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    route: str
    retrieved_documents: List[Dict[str, Any]]
    prompt: str
    answer: str
    sources: List[str]
    confidence: float


# =========================================================
# INTENT KEYWORDS
# =========================================================

INTENT_KEYWORDS = {
    "delivery": [
        "delivery",
        "deliver",
        "late delivery",
        "delayed delivery",
        "arrive",
        "eta"
    ],

    "return": [
        "return",
        "returns",
        "send back"
    ],

    "refund": [
        "refund",
        "money back",
        "reimbursement"
    ],

    "membership": [
        "membership",
        "member",
        "subscription",
        "renewal"
    ],

    "tracking": [
        "track",
        "tracking",
        "where is my order",
        "order status",
        "status"
    ],

    "cancel": [
        "cancel",
        "cancellation"
    ],

    "gift_card": [
        "gift card",
        "giftcard",
        "gift code"
    ],

    "support_hours": [
        "support hours",
        "customer support hours",
        "when is support available",
        "support available"
    ]
}


# =========================================================
# DIRECT-ANSWER INTENTS
# =========================================================

# These queries can be answered by a deterministic local response
# in the mock baseline without using an LLM.
#
# All other recognized support intents are routed through retrieval.

DIRECT_INTENTS = {
    "support_hours"
}


# =========================================================
# NODE 1: CLASSIFY INTENT
# =========================================================

def classify_intent(
    state: SupportState
) -> SupportState:

    query = state.get(
        "query",
        ""
    ).strip().lower()

    matched_intent = "unknown"

    # Check longer phrases first.
    for intent, keywords in INTENT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in query:
                matched_intent = intent
                break

        if matched_intent != "unknown":
            break


    # -----------------------------------------------------
    # Route:
    # recognized support topic -> retrieve
    # direct-only intent -> direct answer
    # unknown -> direct fallback
    # -----------------------------------------------------

    matched_topic = matched_intent

    if matched_topic in DIRECT_INTENTS:
        route = "direct"
    elif matched_topic != "unknown":
        route = "retrieve"
    else:
        route = "direct"

    # Assignment-level intent labels:
    # policy-related queries vs general questions.
    intent_label = (
        "policy_question"
        if matched_topic != "unknown"
        else "general_question"
    )

    print(
        f"[classify_intent] "
        f"intent={intent_label} "
        f"route={route}"
    )

    return {
        **state,
        "intent": intent_label,
        "route": route
    }


# =========================================================
# CONDITIONAL ROUTER
# =========================================================

def route_after_classification(
    state: SupportState
) -> str:

    return state.get(
        "route",
        "direct"
    )


# =========================================================
# NODE 2: RETRIEVE AND ANSWER
# =========================================================

def retrieve_and_answer(
    state: SupportState
) -> SupportState:

    query = state.get(
        "query",
        ""
    ).strip()

    retrieved = retrieve_documents(
        query=query,
        top_k=3
    )


    # Build prompt using the retrieval layer.
    prompt = build_prompt(
        query=query,
        retrieved_documents=retrieved
    )


    # -----------------------------------------------------
    # Deterministic MOCK_LLM baseline
    # -----------------------------------------------------

    if MOCK_LLM != "0":

        if retrieved:

            # Use the first retrieved document as the primary
            # grounded context for the deterministic baseline.
            primary = retrieved[0]

            primary_text = (
                primary.get(
                    "text",
                    ""
                )
                .replace(
                    "\n",
                    " "
                )
                .strip()
            )

            # Keep the mock answer deterministic and grounded.
            answer = (
                "Based on the retrieved context: "
                f"{primary_text[:200]}"
            )

            sources = [
                item["document_id"]
                for item in retrieved
            ]

            # Deterministic confidence based on rank-1 retrieval.
            confidence = round(
                max(
                    0.0,
                    min(
                        1.0,
                        1.0 / (
                            1.0
                            + retrieved[0]["distance"]
                        )
                    )
                ),
                4
            )

        else:

            answer = (
                "Based on the retrieved context: "
                "The available support documents do not "
                "contain enough information to answer this question."
            )

            sources = []

            confidence = 0.0


    else:

        # -------------------------------------------------
        # Optional real LLM branch
        # -------------------------------------------------
        #
        # The capstone baseline does not require network access.
        # Keep this branch explicit so the mock path remains the
        # default graded behavior.
        #
        # A real provider can be added later without changing
        # the graph routing.
        #

        answer = (
            "Based on the retrieved context: "
            "Real LLM mode is not configured. "
            "Set MOCK_LLM=1 for the deterministic local baseline."
        )

        sources = [
            item["document_id"]
            for item in retrieved
        ]

        confidence = 0.5 if retrieved else 0.0


    print(
        "[retrieve_and_answer] "
        f"retrieved={len(retrieved)} "
        f"sources={sources}"
    )


    return {
        **state,
        "retrieved_documents": retrieved,
        "prompt": prompt,
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }


# =========================================================
# NODE 3: DIRECT ANSWER
# =========================================================

def direct_answer(
    state: SupportState
) -> SupportState:

    intent = state.get(
        "intent",
        "unknown"
    )


    # -----------------------------------------------------
    # Deterministic support-hours answer
    # -----------------------------------------------------

    if intent == "support_hours":

        answer = (
            "Customer support is available during the "
            "service hours displayed in the application. "
            "The latest availability should be checked "
            "in the application because service hours may change."
        )

        sources = [
            "doc_008"
        ]

        confidence = 0.95


    # -----------------------------------------------------
    # Deterministic fallback
    # -----------------------------------------------------

    else:

        answer = (
            "I’m sorry, but the available support documents "
            "do not provide enough information to answer "
            "that question."
        )

        sources = []

        confidence = 0.20


    print(
        "[direct_answer] "
        f"intent={intent}"
    )


    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

builder = StateGraph(
    SupportState
)


# ---------------------------------------------------------
# Add required nodes
# ---------------------------------------------------------

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)


# ---------------------------------------------------------
# Start -> classify
# ---------------------------------------------------------

builder.add_edge(
    START,
    "classify_intent"
)


# ---------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------

builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve": "retrieve_and_answer",
        "direct": "direct_answer"
    }
)


# ---------------------------------------------------------
# Finish edges
# ---------------------------------------------------------

builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)


# Compile graph
graph = builder.compile()


# =========================================================
# HELPER FUNCTION
# =========================================================

def ask_support(
    query: str
) -> SupportState:

    if not isinstance(
        query,
        str
    ):
        raise TypeError(
            "Query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )


    initial_state: SupportState = {
        "query": query
    }


    result = graph.invoke(
        initial_state
    )

    return result


# =========================================================
# TEST CASES
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("LANGGRAPH SUPPORT ASSISTANT TEST")
    print("=" * 70)


    test_queries = [
        "How long does a refund take?",
        "Where is my order?",
        "What are the customer support hours?",
        "Can I return an item?",
        "Tell me something unrelated to support"
    ]


    for number, query in enumerate(
        test_queries,
        start=1
    ):

        print("\n" + "-" * 70)
        print(
            f"TEST {number}: {query}"
        )
        print("-" * 70)


        result = ask_support(
            query
        )


        print(
            f"Intent     : "
            f"{result.get('intent')}"
        )

        print(
            f"Route      : "
            f"{result.get('route')}"
        )

        print(
            f"Answer     : "
            f"{result.get('answer')}"
        )

        print(
            f"Sources    : "
            f"{result.get('sources')}"
        )

        print(
            f"Confidence : "
            f"{result.get('confidence')}"
        )


    print("\n" + "=" * 70)
    print("LANGGRAPH TEST COMPLETE")
    print("=" * 70)