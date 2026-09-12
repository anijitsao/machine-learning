import os

from dotenv import load_dotenv
from fastapi import FastAPI

from src.routes import index_router, prediction_router

load_dotenv()
app = FastAPI(
    description=os.getenv("APP_DESCRIPTION", "description"),
    title=os.getenv("APP_TITLE", "Title"),
)


app.include_router(router=index_router)
app.include_router(router=prediction_router, prefix="/prediction")
