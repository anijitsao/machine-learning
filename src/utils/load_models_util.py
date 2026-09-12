# This will load ML models from PKL file to FastAPI
# app.state in the lifespan event i.e server startup / shutdown

from pathlib import Path

from joblib import load


def load_ml_models():
    try:
        models = {}
        base_path = Path("./")
        models["classification"] = load(
            base_path.joinpath("./", "models/classification.pkl")
        )
        models["regression"] = load(base_path.joinpath("models", "regression.pkl"))
    except FileExistsError as e:
        print("Unable to load models", e)
    return models
