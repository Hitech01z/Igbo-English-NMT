from pydantic import BaseModel


class TranslationRequest(BaseModel):
    text: str
    direction: str


class TranslationResponse(BaseModel):
    translation: str