from typing import List

from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    """
    Structured response returned by the support assistant.
    """

    answer: str = Field(
        min_length=1,
        description="Final answer for the customer."
    )

    sources: List[str] = Field(
        description="Retrieved document IDs used for the answer."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1."
    )


def validate_support_response(
    answer: str,
    sources: List[str],
    confidence: float
) -> SupportResponse:

    return SupportResponse(
        answer=answer,
        sources=sources,
        confidence=confidence
    )


if __name__ == "__main__":

    print("=" * 70)
    print("PYDANTIC SUPPORT RESPONSE TEST")
    print("=" * 70)

    # Valid response
    valid_response = validate_support_response(
        answer="Your refund is normally initiated within 10 working days after approval.",
        sources=["doc_006"],
        confidence=0.92
    )

    print("\nValid response:")
    print(valid_response.model_dump())


    # Invalid confidence test
    print("\n" + "-" * 70)
    print("INVALID CONFIDENCE TEST")
    print("-" * 70)

    try:
        validate_support_response(
            answer="Test answer",
            sources=["doc_001"],
            confidence=1.5
        )

    except Exception as error:
        print(
            "Validation correctly rejected the invalid value:"
        )
        print(error)


    # Invalid empty answer test
    print("\n" + "-" * 70)
    print("INVALID ANSWER TEST")
    print("-" * 70)

    try:
        validate_support_response(
            answer="",
            sources=[],
            confidence=0.5
        )

    except Exception as error:
        print(
            "Validation correctly rejected the empty answer:"
        )
        print(error)


    print("\n" + "=" * 70)
    print("PYDANTIC TEST COMPLETE")
    print("=" * 70)