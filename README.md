# Iris Multiclass Classification

A machine learning project that classifies Iris flowers into three species using Python and Logistic Regression.

The model uses four flower measurements to predict the species:
1 Sepal length
2 Sepal width
3 Petal length
4 Petal width

The three target classes are:
1 Setosa
2 Versicolor
3 Virginica

# Technologies Used
1 python 
2 pandas
3 scikit learn
4 joblib

# Model Performance
The model was evaluated on a test set containing 30 samples.

1 Model: Logistic Regression
2 Accuracy: 96.67%
3 Correct predictions: 29 out of 30

The model performed well on this test split. However, this result alone does not guarantee the same performance on all unseen data.

# Project Structure

iris_multiclass_project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── iris_model.joblib
├── notebooks/
├── src/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── predict.py
│   └── test_prediction.py
├── tests/
├── README.md
└── .gitignore

# how to run

# 1  Clone the repository

git clone <repository-url>
cd iris_multiclass_project

2 Create a virtual environment

python -m venv .venv

3- Activate the virtual environment

On Windows PowerShell:
.venv\Scripts\Activate.ps1

4- Install the required libraries

python -m pip install -r requirements.txt

5- Run the project
Inspect the dataset:
python src/inspect_data.py

Train and evaluate the model:
python src/train.py

Make a prediction:
python src/predict.py

Run the prediction tests:
python src/test_prediction.py

Git practice: learning version control step by step.
Learning Git branches: feature-test.