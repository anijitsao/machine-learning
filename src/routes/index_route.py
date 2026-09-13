from fastapi import APIRouter

from src.models import IndexResponseModel

router = APIRouter(tags=["index"])


@router.get("/", responses={200: {"model": IndexResponseModel}})
def index_route():
    return {"message": "Index route reached"}
