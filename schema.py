from pydantic import BaseModel,Field
from typing import Literal


# create a root response object
class RootResponse(BaseModel):
    """creates the payload for the root endpoint."""
    message: str = Field(..., description="root message",
                         examples=["we are live!"])
    

# create a model response object
class ModelResponse(BaseModel):
    """
    creates a response object for the model prediction.
    """
    predicted_charges : float = Field(..., 
                                      description= "The model's predicted charges",
                                      gt= 0, examples=[55.3, 44.2])

# age     sex     bmi  chilregion     region
# create the model request object
class ModelRequest(BaseModel):
    """
    creates the request object for the model prediction
    """
    age: int = Field(..., description="Age of the client",
                     gt=5, lt=100, examples=[35])
    sex: Literal['Male','Female'] = Field(..., description="Gender of client",
                                          examples=["Male","Female"])
    bmi: float = Field(..., description="bmi of the client",
                     gt=5, lt=100, examples=[35.4])  
    
    children: int = Field(..., description="no children of the client",
                     ge=0, lt=20, examples=[3])
    smoker: Literal['Yes','No'] = Field(..., description="smoking habit of client",
                                          examples=["Yes","No"])
    region: Literal['southeast','southwest',
                    'northeast', 'northwest'] = Field(..., description="region of the client",
                                          examples=['southeast','southwest'])