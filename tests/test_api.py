from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json() == {
        "message": "Iris API is running!"
    }
    
def test_predict_setosa():
    response = client.post("/predict", json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },)
    assert response.status_code == 200
    assert response.json()["predicted_class"]=="setosa"
    assert response.json()["class_id"] == 0
    
def test_predict_invalid_input():
    response = client.post("/predict" , json={
            "sepal_length": "invalid",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
    },)
    assert response.status_code == 422 