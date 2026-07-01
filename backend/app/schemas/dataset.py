from pydantic import BaseModel
from typing import Dict


class DatasetStatistics(BaseModel):

    total_sentences: int

    domains: Dict[str, int]

    duplicates: int

    missing_values: int