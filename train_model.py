import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Dataset load karna
data = pd.read_csv("dataset/diabetes.csv")

# Features aur target split
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model train karna
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Accuracy print karna
print("Model Accuracy:", model.score(X_test, y_test))

# Model save karna
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully as diabetes_model.pkl")
