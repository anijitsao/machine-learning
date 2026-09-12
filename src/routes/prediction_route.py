from fastapi import APIRouter, Request

from src.models import PredictDiabetesModel, PredictWineModel
from src.services import get_diabetes_prediction, get_wine_type_prediction

router = APIRouter(tags=["prediction"])


@router.post("/wine-types")
def predict_wine_types(payload: PredictWineModel, req: Request):
    return get_wine_type_prediction(payload=payload, req=req)


@router.post("/diabetes-progression")
def predict_diabetes_progression(payload: PredictDiabetesModel, req: Request):
    return get_diabetes_prediction(payload=payload, req=req)
