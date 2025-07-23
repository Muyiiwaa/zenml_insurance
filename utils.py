from zenml.client import Client
from sklearn.ensemble import RandomForestRegressor
import logging
from logging import getLogger
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os

load_dotenv()

model_id = os.getenv(key="MODEL_ARTIFACT")
scaler_id = os.getenv(key="SCALER_ARTIFACT")
encoder_id = os.getenv(key="ENCODER_ARTIFACT")

# configure logging
logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

# get the columns in the training_data
def get_columns() -> List[str]:
    artifact = Client().get_artifact_version("1e464976-0435-4ebf-afec-b6b2d5cdd062")
    data = artifact.load()
    column_names = list(data.columns)
    
    return column_names

# get the scaler, encoder and model objects.
def get_artifacts() -> Tuple[Dict, StandardScaler, RandomForestRegressor]:
    """This function returns the encoder dictionary,
    model, standard scaler object and model artifact
    associated with the prod training pipeline."""
    
    # encoder
    artifact = Client().get_artifact_version(str(encoder_id))
    label_encoders = artifact.load()
    # scaler
    artifact = Client().get_artifact_version(str(scaler_id))
    scaler = artifact.load()
    # model
    artifact = Client().get_artifact_version(str(model_id))
    model = artifact.load()
    
    return label_encoders, scaler, model


# write a function that makes prediction

def predict_charges(data: Dict) -> float:
    final_data = {column:[value] for column, value in data.items()}
    logger.info(f"data: {final_data}")
    final_data = pd.DataFrame(final_data)
    label_encoders, scaler, model = get_artifacts()
    for column_name in label_encoders.keys():
        final_data[column_name] = label_encoders[column_name].transform(final_data[column_name])
    # scale the dataset
    columns = list(final_data.columns)
    final_data = scaler.transform(final_data)
    final_data = pd.DataFrame(data=final_data, columns=columns)
    
    # get prediction
    pred = model.predict(final_data)
    
    return pred[0]


if __name__ == "__main__":
    data = {"age": 35,
            "sex": "male",
            "bmi": 35.4,
            "children": 3,
            "smoker": "yes",
            "region": "southeast"}
    print(predict_charges(data=data))



