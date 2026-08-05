from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Metadata sent along with a prediction request.

    The image itself will be uploaded using multipart/form-data,
    so this model is mainly for optional information.
    """

    user_id: str | None = Field(
        default=None,
        description="Unique user identifier"
    )

    age: int | None = Field(
        default=None,
        ge=0,
        le=120,
        description="User age"
    )

    gender: str | None = Field(
        default=None,
        description="User gender"
    )

    skin_type: str | None = Field(
        default=None,
        description="Skin type (Oily, Dry, Combination, Sensitive)"
    )

    symptoms: str | None = Field(
        default=None,
        description="Additional symptoms provided by the user"
    )