from fastapi import APIRouter, Request

from src.models import (
    DiabetesResponseModel,
    ErrorModel,
    PredictDiabetesModel,
    PredictWineModel,
    WineResponseModel,
)
from src.services import get_diabetes_prediction, get_wine_type_prediction

router = APIRouter(tags=["prediction"])


@router.post(
    "/wine-types",
    responses={
        200: {"model": WineResponseModel, "description": "Successful Response"},
        400: {"model": ErrorModel, "description": "Invalid Input Data"},
        500: {"model": ErrorModel, "description": "Internal Server Error"},
    },
)
def predict_wine_types(payload: PredictWineModel, req: Request):
    return get_wine_type_prediction(payload=payload, req=req)


@router.post(
    "/diabetes-progression",
    responses={
        200: {"model": DiabetesResponseModel, "description": "Successful Response"},
        400: {"model": ErrorModel, "description": "Invalid Input Data"},
        500: {"model": ErrorModel, "description": "Internal Server Error"},
    },
)
def predict_diabetes_progression(payload: PredictDiabetesModel, req: Request):
    return get_diabetes_prediction(payload=payload, req=req)
