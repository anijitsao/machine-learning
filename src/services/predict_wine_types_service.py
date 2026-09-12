import pandas as pd
from fastapi import HTTPException, Request


def predict_wine_type(payload, req: Request):
    try:
        # conversion to dictinary and modifying attributes
        payload_dict = payload.dict()
        payload_dict["od280/od315_of_diluted_wines"] = payload_dict["od315_of_diluted_wines"]
        del payload_dict["od315_of_diluted_wines"]

        # convert to pandas dataframe. Othewise feature names will be missed and ValueError
        payload_reshaped = pd.DataFrame([payload_dict])
        predictions = req.app.state.ml_models["classification"].predict(payload_reshaped)
        
        # convert the predictions to serialize for JSON object
        predicted_type = int(predictions[0])
        return {"wine_type": predicted_type}
    except Exception as e:  # noqa: BLE001
        print("Error occurred while getting predictions", e)
        return HTTPException(500, e)    
