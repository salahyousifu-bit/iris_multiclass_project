from sklearn.datasets import load_iris 
import pandas as pd 
from sklearn.model_selection import train_test_split

iris = load_iris()

df = pd.DataFrame(iris.data , columns=iris.feature_names)

df["target"]=iris.target 

X = df.drop(columns=["target"])
y = df["target"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

print("training features :",X_train.shape)
print("testing features  :", X_test.shape)
print("training taple    :",y_train.shape)
print("testinig taple    :",y_test.shape)

print("\ntraining class count :")
print(y_train.value_counts().sort_index())
print("\n testing clss count  :")
print(y_test.value_counts().sort_index())

