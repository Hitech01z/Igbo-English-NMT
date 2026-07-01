from fastapi import APIRouter
from app.schemas.contribution import ContributionRequest
from app.services.contribution_service import save_contribution

router = APIRouter()

@router.post("/")
def contribute(data: ContributionRequest):

    return save_contribution(data)