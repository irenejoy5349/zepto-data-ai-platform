from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from support_assistant.graph import ask_support


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description="Local RAG-based customer support assistant",
    version="1.0.0",
)


# =========================================================
# REQUEST MODEL
# =========================================================

class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Customer support question.",
    )


# =========================================================
# RESPONSE MODEL
# =========================================================

class AskResponse(BaseModel):
    answer: str = Field(
        min_length=1
    )

    sources: List[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "status": "running",
    }


# =========================================================
# ASK ENDPOINT
# =========================================================

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(
    request: AskRequest
):
    try:

        result = ask_support(
            request.query
        )

        return AskResponse(
            answer=result.get(
                "answer",
                ""
            ),
            sources=result.get(
                "sources",
                []
            ),
            confidence=float(
                result.get(
                    "confidence",
                    0.0
                )
            ),
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {error}"
        )