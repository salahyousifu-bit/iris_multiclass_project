from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="Iris Classification API")

project_dir = Path(__file__).resolve().parent.parent
path_model = project_dir / "models" / "iris_model.joblib"
model = joblib.load(path_model) 

class_names = ["setosa", "versicolor", "virginica"]

class FlowerFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
@app.get("/")
def home():
    return {"message": "Iris API is running!"}

@app.post("/predict")
def predict(features:FlowerFeatures):
    flower = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    prediction = int(model.predict(flower)[0])
    
    return {
        "predicted_class":class_names[prediction],
        "class_id":prediction,
    }