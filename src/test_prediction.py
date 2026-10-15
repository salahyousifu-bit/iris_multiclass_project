import joblib 
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
path_model = project_dir / "models" / "iris_model.joblib"

assert path_model.exists()

model = joblib.load(path_model)

new_flower = [[5.1, 3.5, 1.4, 0.2]]

result = model.predict(new_flower)

assert len(result) == 1
assert result[0] == 0

print("all tests passed")