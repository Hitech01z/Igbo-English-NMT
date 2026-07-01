from fastapi import APIRouter

from app.services.dataset_service import (
    get_dataset,
    dataset_statistics
)

router = APIRouter()


@router.get("/")
def dataset():

    return get_dataset()


@router.get("/statistics")
def statistics():

    return dataset_statistics()