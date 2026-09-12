import numpy as np
from fastapi import HTTPException, Request


def predict_wine_type(payload, req: Request):
    try:
        payload_reshaped = np.array(payload).reshape(-1,1)
        predictions = req.app.state.ml_models["classification"].predict(payload_reshaped)
        print("predictions", predictions)
        return {"wine_type": 2}
    except Exception as e:  # noqa: BLE001
        print("Error occurred while getting predictions", e)
        return HTTPException(500, e)    
