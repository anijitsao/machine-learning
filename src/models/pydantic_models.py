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