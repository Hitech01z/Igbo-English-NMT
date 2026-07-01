from pydantic import BaseModel


class DashboardResponse(BaseModel):

    total_sentences: int

    education: int

    translations: int

    bleu_score: float