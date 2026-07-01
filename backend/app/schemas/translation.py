from pydantic import BaseModel


class TranslationRequest(BaseModel):

    text: str

    source_language: str

    target_language: str


class TranslationResponse(BaseModel):

    input: str

    translation: str