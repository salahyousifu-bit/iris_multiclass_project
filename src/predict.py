import joblib
from pathlib import Path
from sklearn.datasets  import load_iris

project_root = Path(__file__).resolve().parent.parent
path_model = project_root / "models" / "iris_model.joblib"

model = joblib.load(path_model)


new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)

iris = load_iris()

result = iris.target_names[prediction[0]]


print("predicted class number :" , prediction[0])
print("predicted class name   :",result)