from pydantic import BaseModel, Field, field_validator


class QuestionPair(BaseModel):
    question1: str = Field(..., min_length=3, max_length=2000)
    question2: str = Field(..., min_length=3, max_length=2000)

    @field_validator("question1", "question2")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty")
        return value
