from fastapi import APIRouter

from app.schemas.translation import TranslationRequest

from app.services.translation_service import translate


router = APIRouter()


@router.post("/")

def translate_endpoint(data: TranslationRequest):

    result = translate(

        text=data.text,

        source_language=data.source_language,

        target_language=data.target_language

    )

    return {

        "input": data.text,

        "translation": result

    }