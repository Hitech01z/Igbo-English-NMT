from inference.dual_translator import translate


def translate_text(
    text,
    source_language,
    target_language,
):

    return translate(

        text=text,

        source_language=source_language,

        target_language=target_language

    )