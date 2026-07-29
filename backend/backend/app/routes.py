from fastapi import APIRouter

from app.schemas import (
    TranslationRequest,
    TranslationResponse,
)

from app.translator import translate_text

router = APIRouter()


@router.get("/")
def health():
    return {
        "status": "running",
        "model": "Round 2 Transformer"
    }


@router.post("/translate")
def translate(request: TranslationRequest):

    source, target = request.direction.split("-")

    prediction = translate_text(
        text=request.text,
        source_language=source,
        target_language=target,
    )

    return TranslationResponse(
        translation=prediction
    )