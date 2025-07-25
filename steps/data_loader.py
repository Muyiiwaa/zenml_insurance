# import the needed libraries

from zenml import step
import pandas as pd
import numpy as np
from zenml.logger import get_logger
from typing_extensions import Annotated
from typing import Optional

# configure our logging
logger = get_logger(__name__)

# define the step

@step
def load_data() -> Annotated[Optional[pd.DataFrame], "Full DataSet"]:
    """
    This step loads the full dataset needed for the model training pipeline.
    """
    data = None
    try:
        data = pd.read_csv(r'.\insurance.csv')
        logger.info(f"""
                    Loaded the dataframe successfully.
                    shape: {data.shape}.
                    """)
    except Exception as err:
        logger.error(f'An error occured. Detail: {err}')
    
    return data

if __name__ == "__main__":
    pd.read_csv(r'.\insurance.csv')