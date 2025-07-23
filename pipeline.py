import pandas as pd
from zenml import pipeline
from steps.data_loader import load_data
from steps.data_prep import encode_step,split_dataset,scale_dataset
from steps.model_training import train_model
from zenml.logger import get_logger


logger = get_logger(__name__)


# define a pipeline function
@pipeline(enable_cache=False)
def insurance_pipeline(enable_cache=False):
    data = load_data()
    data, label_encoders = encode_step(data)
    X_train, X_test, y_train, y_test = split_dataset(data)
    X_train, X_test, scaler = scale_dataset(X_train, X_test)
    model = train_model(X_train, y_train, X_test, y_test)
    

if __name__ == "__main__":
    insurance_pipeline()