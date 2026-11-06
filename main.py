from fastapi import FastAPI
from pydantic import BaseModel
from predict import predicted

# How to build a server with FastApi 
# create fastapi application
app = FastAPI(title="FastAPI Basic Demo")
    
# mock ML Prediction endpoint 

class PredictionInput(BaseModel):
    Pregnancies:int
    Glucose:int
    BloodPressure:int
    SkinThickness:int
    Insulin:float
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int
    
@app.post("/predict")
def Predict_diabetes(input_data:PredictionInput):
    prediction=predicted(input_data.model_dump())
    return {
        "Prediction":int(prediction)
        
    }        