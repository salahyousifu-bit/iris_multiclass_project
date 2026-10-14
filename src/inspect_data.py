import pandas as pd 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris = load_iris()

df = pd.DataFrame(iris.data , columns=iris.feature_names)
df["target"]=iris.target

#display the first five rows
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\n missing varues : ")
print(df.isnull().sum())
print("\n target class counts : ")
print(df["target"].value_counts().sort_index())
print("\n Statistical summary :")
print(df.describe())
print("\n data types :")
print(df.dtypes)
print("\n Mean measurements for each flower class :")
print(df.groupby("target").mean(numeric_only=True))
print("\n Target class names :")
for class_id , class_name in enumerate(iris.target_names):
    print(f"{class_id}:{class_name}")
    

mean_petal_length = df.groupby("target")["petal length (cm)"].mean()


class_names = iris.target_names

plt.bar(class_names , mean_petal_length.values)
plt.title("Mean Petal Length by Iris Class")
plt.xlabel("Flower Class")
plt.ylabel("Mean Petal Length (cm)")
plt.tight_layout()
plt.show()