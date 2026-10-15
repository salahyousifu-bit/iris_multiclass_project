from pathlib import Path
import joblib

def load_model():
    project_dir = Path(__file__).resolve().parent.parent
    path_model = project_dir / "models" / "iris_model.joblib"
    
    model =  joblib.load(path_model)
    assert model
    return model

def test_prediction_returns_setosa():
    model = load_model()
    
    new_flower = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(new_flower)
    
    assert prediction[0] == 0
    
    
def test_prediction_returns_valid_class():
    model = load_model()
    
    new_flower = [[5.1, 3.5, 1.4, 0.2]]  
    predict = model.predict(new_flower)
    
    assert predict[0] in [0,1,2]
    
def test_prediction_handles_multiple_flowers():
    model = load_model()
    flowers = [
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.9, 4.5, 1.5],
        [6.5, 3.0, 5.2, 2.0],
    ]
    predictions = model.predict(flowers)
    
    assert len(predictions) == 3 
    assert all(prediction in [0,1,2] for prediction in predictions)