from fastapi import APIRouter

from app.services.statistics_service import (
    dashboard_statistics
)

router = APIRouter()


@router.get("/")

def dashboard():

    return dashboard_statistics()