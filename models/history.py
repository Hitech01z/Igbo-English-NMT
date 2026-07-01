from pydantic import BaseModel


class TranslationHistory(BaseModel):

    source_text: str

    translated_text: str

    source_language: str

    target_language: str