import os

from dotenv import load_dotenv
from fastapi import FastAPI

from src.routes import index_router, prediction_router
from src.utils import load_ml_models
from contextlib import asynccontextmanager

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.ml_models = load_ml_models()
        print("ML model is loaded")
    except FileNotFoundError as e:
        print(f"failed to load models {e}")
    yield
    # shutdown clean up
    app.state.ml_models = None


app = FastAPI(
    description=os.getenv("APP_DESCRIPTION", "description"),
    title=os.getenv("APP_TITLE", "Title"),
    lifespan=lifespan,
)
app.include_router(router=index_router)
app.include_router(router=prediction_router, prefix="/prediction")
