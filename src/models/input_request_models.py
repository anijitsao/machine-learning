from pydantic import BaseModel


class PredictWineModel(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenol: float
    flavanoids: float
    nonflavanoids_phenols: float
    proanthrocyanins: float
    color_intensity: float
    hue: float
    od315_of_diluted_wines: float
    proline: float

class PredictDiabetesModel(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float