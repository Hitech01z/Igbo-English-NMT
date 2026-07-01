from fastapi import APIRouter

from app.services.translation_service import translate_text

router = APIRouter()

@router.post("/translate")

def translate(data: dict):

    result = translate_text(

        data["text"]

    )

    return result