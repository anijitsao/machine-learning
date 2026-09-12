from fastapi import APIRouter

router = APIRouter(tags=["prediction"])

@router.post("/wine-types")
def predict_wine_types():
    return {"message": "Prediction route reached"}


@router.post("/diabetes-progression")
def predict_diabetes_progression():
    return {"message": "Prediction diabetes route reached"}