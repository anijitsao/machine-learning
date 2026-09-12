import os

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
app = FastAPI(
    description=os.getenv("APP_DESCRIPTION", "description"),
    title=os.getenv("APP_TITLE", "Title"),
)


@app.get("/")
def index():
    return {"message": "Index route reached"}
