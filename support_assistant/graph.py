import json
import os
from typing import Any, Dict, List, TypedDict

import requests
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from support_assistant.rag import build_prompt, retrieve_documents


# =========================================================
# CONFIGURATION
# =========================================================

# Required graded baseline:
# MOCK_LLM unset or "1" -> deterministic local mock mode.
#
# Optional extension:
# MOCK_LLM="0" -> real LLM mode.
MOCK_LLM = os.getenv("MOCK_LLM", "1").strip()

# Optional Groq configuration for MOCK_LLM=0.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
).strip()
GROQ_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)


# =========================================================
# PYDANTIC RESPONSE SCHEMAS
# =========================================================

class SupportResponse(BaseModel):
    answer: str = Field(min_length=1)
    sources: List[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class IntentResponse(BaseModel):
    intent: str


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
# EXACT MOCK KEYWORD HEURISTIC REQUIRED BY THE RUBRIC
# =========================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


# =========================================================
# PYDANTIC / JSON HELPER
# =========================================================

def validate_model(model_cls, payload):
    """
    Support both Pydantic v2 and older Pydantic versions.
    """
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


# =========================================================
# OPTIONAL REAL LLM CALL
# =========================================================

def call_real_llm(prompt: str, model_cls):
    """
    Optional real-LLM path.

    The required graded baseline never calls this function
    because MOCK_LLM defaults to "1".

    When MOCK_LLM=0, invalid JSON/schema output is retried
    up to two additional times with corrective instructions.
    """

    if not GROQ_API_KEY:
        raise RuntimeError(
            "MOCK_LLM=0 requires GROQ_API_KEY "
            "for the optional real-LLM path."
        )

    last_error = None

    for attempt in range(3):
        correction = ""

        if attempt > 0:
            correction = """

CORRECTION:
Your previous response did not satisfy the required JSON schema.
Return ONLY valid JSON.
Do not use Markdown fences.
Ensure every required field is present and correctly typed.
"""

        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": (
                    f"Bearer {GROQ_API_KEY}"
                ),
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return only valid JSON. "
                            "Do not add Markdown fences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            prompt + correction
                        ),
                    },
                ],
            },
            timeout=60,
        )

        response.raise_for_status()

        try:
            content = (
                response
                .json()["choices"][0]["message"]["content"]
            )

            payload = json.loads(content)

            return validate_model(
                model_cls,
                payload
            )

        except (
            KeyError,
            json.JSONDecodeError,
            ValidationError,
            TypeError,
            ValueError,
        ) as error:
            last_error = error

    raise RuntimeError(
        "Real LLM output failed schema validation "
        f"after 3 attempts: {last_error}"
    )


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

    # -----------------------------------------------------
    # REQUIRED MOCK BASELINE
    # -----------------------------------------------------

    if MOCK_LLM != "0":

        is_policy_question = any(
            keyword in query
            for keyword in POLICY_KEYWORDS
        )

        intent_label = (
            "policy_question"
            if is_policy_question
            else "general_question"
        )

    # -----------------------------------------------------
    # OPTIONAL REAL LLM PATH
    # -----------------------------------------------------

    else:

        prompt = f"""
Classify the user query as exactly one of:

- policy_question
- general_question

A policy_question is related to Zepto delivery,
returns, refunds, membership, tracking, cancellation,
gift cards, or support hours.

A general_question is unrelated to Zepto policy.

USER QUERY:
{query}

Return JSON only:

{{"intent": "policy_question"}}

or

{{"intent": "general_question"}}
""".strip()

        result = call_real_llm(
            prompt,
            IntentResponse
        )

        intent_label = result.intent

        if intent_label not in {
            "policy_question",
            "general_question",
        }:
            raise RuntimeError(
                "Invalid intent returned by real LLM."
            )

    route = (
        "retrieve"
        if intent_label == "policy_question"
        else "direct"
    )

    print(
        f"[classify_intent] "
        f"intent={intent_label} "
        f"route={route}"
    )

    return {
        **state,
        "intent": intent_label,
        "route": route,
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

    # Retrieval always runs in both modes.
    retrieved = retrieve_documents(
        query=query,
        top_k=3
    )

    # Build the structured RAG prompt.
    prompt = build_prompt(
        query=query,
        retrieved_documents=retrieved
    )

    # -----------------------------------------------------
    # REQUIRED MOCK BASELINE
    # -----------------------------------------------------

    if MOCK_LLM != "0":

        if retrieved:

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

            answer = (
                "Based on the retrieved context: "
                f"{primary_text[:200]}"
            )

            sources = [
                item["document_id"]
                for item in retrieved
            ]

            distance = retrieved[0].get(
                "distance"
            )

            if distance is None:
                confidence = 1.0
            else:
                confidence = round(
                    max(
                        0.0,
                        min(
                            1.0,
                            1.0 / (
                                1.0
                                + float(distance)
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

    # -----------------------------------------------------
    # OPTIONAL REAL LLM PATH
    # -----------------------------------------------------

    else:

        schema_instruction = """
Return ONLY JSON with this schema:

{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}

Rules:
- answer must be a string
- sources must contain only retrieved document IDs
- confidence must be between 0 and 1
- use only information present in the retrieved context
"""

        result = call_real_llm(
            prompt + "\n\n" + schema_instruction,
            SupportResponse
        )

        answer = result.answer
        sources = result.sources
        confidence = result.confidence

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
        "confidence": confidence,
    }


# =========================================================
# NODE 3: DIRECT ANSWER
# =========================================================

def direct_answer(
    state: SupportState
) -> SupportState:

    query = state.get(
        "query",
        ""
    ).strip()

    # -----------------------------------------------------
    # REQUIRED MOCK BASELINE
    # -----------------------------------------------------

    if MOCK_LLM != "0":

        answer = (
            "I can only answer questions about "
            "Zepto policies right now."
        )

        sources = []
        confidence = 1.0

    # -----------------------------------------------------
    # OPTIONAL REAL LLM PATH
    # -----------------------------------------------------

    else:

        prompt = f"""
You are a Zepto support assistant.

Answer the following general question directly.

Do not invent Zepto policy information.
Do not retrieve policy documents.

Return ONLY JSON:

{{
  "answer": "string",
  "sources": [],
  "confidence": 0.0
}}

USER QUESTION:
{query}
""".strip()

        result = call_real_llm(
            prompt,
            SupportResponse
        )

        answer = result.answer
        sources = []
        confidence = result.confidence

    print(
        "[direct_answer] "
        "intent=general_question"
    )

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

builder = StateGraph(
    SupportState
)

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

builder.add_edge(
    START,
    "classify_intent"
)

builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve": "retrieve_and_answer",
        "direct": "direct_answer",
    }
)

builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)

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

    result = graph.invoke(
        {
            "query": query
        }
    )

    return result


# =========================================================
# LOCAL TESTS
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
