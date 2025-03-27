# File: model_training/train_model.py
# ----------------------------------------------------
# Trains an ML model to predict required specialists for patients
# based on symptoms, disease history, and other medical data.
# ----------------------------------------------------

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
import joblib

# Load training data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").astype(int)  # Convert labels to integers
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").astype(int)

# Ensure 'Doctors_Required' is not in features (to prevent data leakage)
if "Doctors_Required" in X_train.columns:
    X_train = X_train.drop(columns=["Doctors_Required"])
    X_test = X_test.drop(columns=["Doctors_Required"])

# Normalize numerical features
scaler = StandardScaler()
numeric_cols = ["Age", "Consultation_Duration_Minutes", "BMI", "Past_Consultations"]
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Reduce model overfitting using regularization
base_model = RandomForestClassifier(
    n_estimators=50,  # Reduce number of trees
    max_depth=4,  # Lower max depth to prevent memorization
    min_samples_split=15,  # Require at least 15 samples to split a node
    min_samples_leaf=7,  # Require at least 7 samples per leaf
    random_state=42,
    class_weight="balanced_subsample"  # Handle class imbalance
)

model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

# Perform cross-validation to evaluate generalization
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_micro')
print(f"Cross-Validation F1 Score: {cv_scores.mean():.2f}")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "trained_appointment_model.pkl")
print("Model training complete. Saved as 'trained_appointment_model.pkl'")
