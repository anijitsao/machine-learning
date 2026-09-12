from fastapi import APIRouter

router = APIRouter(tags=["index"])

@router.get("/")
def index_route():
    return {"message": "Index route reached"}