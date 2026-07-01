from pydantic import BaseModel
from typing import Optional


class ContributionRequest(BaseModel):

    english: str

    igbo: str

    domain: str

    source: str

    contributor: str


class ContributionResponse(BaseModel):

    success: bool

    message: str

    id: Optional[int] = None