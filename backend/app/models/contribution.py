from pydantic import BaseModel

class Contribution(BaseModel):

    igbo: str

    english: str

    domain: str

    contributor: str