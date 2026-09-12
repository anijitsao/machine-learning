from fastapi import APIRouter, Request

from src.models import PredictWineModel
from src.services import predict_wine_type

router = APIRouter(tags=["prediction"])

@router.post("/wine-types")
def predict_wine_types(payload: PredictWineModel, req: Request):
    return predict_wine_type(payload=payload, req=req)


@router.post("/diabetes-progression")
def predict_diabetes_progression():
    return {"message": "Prediction diabetes route reached"}