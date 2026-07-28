from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.dashboard import router as dashboard_router
from app.dataset import router as dataset_router
from app.contribution import router as contribution_router

from app.routes import router


app = FastAPI(
    title="Igbo-English NMT API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # we'll restrict this later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)
app.include_router(dashboard_router)
app.include_router(dataset_router)
app.include_router(contribution_router)