from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.translate import router as translate_router
from app.api.dashboard import router as dashboard_router
from app.api.dataset import router as dataset_router
from app.api.contribute import router as contribute_router
from app.api.history import router as history_router
from app.api.dataset_explorer import (
    router as dataset_explorer_router)
from app.api.admin import router as admin_router

import os


app = FastAPI(
    title="Igbo-English NMT API",
    version="1.0"
)


app.include_router(
    translate_router,
    prefix="/translate",
    tags=["Translation"]
)


app.include_router(

    contribute_router,

    prefix="/contribute",

    tags=["Contribution"]

)

app.include_router(
    history_router,
    prefix="/history",
    tags=["History"]
)

app.include_router(
    dataset_explorer_router,
    prefix="/dataset-explorer",
    tags=["Dataset Explorer"]
)

app.include_router(
    admin_router,
    prefix="/admin",
    tags=["Admin"]
)

app.include_router(
    dashboard_router,
    prefix="/dashboard",
    tags=["Dashboard"]
)

app.include_router(
    dataset_router,
    prefix="/dataset",
    tags=["Dataset"]
)

origins = os.getenv(
    "ALLOWED_ORIGINS",
    "*"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():

    return {

        "project": "Igbo-English Neural Machine Translation",

        "method": "Iterative Back Translation",

        "status": "running"

    }