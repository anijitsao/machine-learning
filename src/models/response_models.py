from pydantic import BaseModel


class WineModel(BaseModel):
    wine_type: int


class WineResponseModel(BaseModel):
    data: WineModel


class DiabetesModel(BaseModel):
    diabetes_progression_score: float

class DiabetesResponseModel(BaseModel):
    data: DiabetesModel

class IndexModel(BaseModel):
    message: str

class IndexResponseModel(BaseModel):
    data: IndexModel        