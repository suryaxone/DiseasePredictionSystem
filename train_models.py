import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create model directory
os.makedirs("models", exist_ok=True)

# ---------------- Diabetes ----------------
diabetes = pd.read_csv("dataset/diabetes.csv")
X_d = diabetes.drop(columns=['Outcome'])
y_d = diabetes['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

rf_d = RandomForestClassifier(n_estimators=100, random_state=42)
rf_d.fit(X_train, y_train)

with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump(rf_d, f)

print("✅ Diabetes model saved!")

# ---------------- Heart ----------------
heart = pd.read_csv("dataset/heart.csv")
X_h = heart.drop(columns=['target'])
y_h = heart['target']

X_train, X_test, y_train, y_test = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

rf_h = RandomForestClassifier(n_estimators=100, random_state=42)
rf_h.fit(X_train, y_train)

with open("models/heart_model.pkl", "wb") as f:
    pickle.dump(rf_h, f)

print("✅ Heart model saved!")
