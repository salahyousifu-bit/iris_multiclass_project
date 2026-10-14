from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
import joblib
from pathlib import Path

iris = load_iris()
df = pd.DataFrame(iris.data , columns=iris.feature_names)

X = iris.data 

y = iris.target 

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

model = LogisticRegression()

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

print(y_pred)

accuracy = accuracy_score(y_test , y_pred)
print("accuracy :" , accuracy)

print("\nconfusion matrix :")
print(confusion_matrix(y_pred , y_test))

print("target names")
print(iris.target_names)

print("\nclassification report")
print(classification_report(y_pred , y_test , target_names=iris.target_names))

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "iris_model.joblib"

joblib.dump(model , model_path)

print("\nmodel saved successfuly ")
print("model path ", model_path)
print("File exists:", model_path.exists())