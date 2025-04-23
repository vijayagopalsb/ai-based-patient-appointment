[Back to Main README](../README.md)

---

## Model Training Guide: AI-based Patient Appointment System

This document explains the **model training pipeline** used in the project, including the algorithm, hyperparameter tuning, evaluation metrics, and output artifacts.

---

## Model Architecture

The project employs a **multi-label classification model** to predict the required doctors for each patient based on their health profile.

- **Base Model**: `RandomForestClassifier`

- **Multi-label Handling**: `MultiOutputClassifier` (wraps RandomForest to support multi-label outputs)

---

## ⚙️ Hyperparameter Tuning

The model uses **GridSearchCV** for hyperparameter tuning with **5-fold cross-validation**.

### 🔍 Parameter Grid:

| Parameter                | Values                |
|--------------------------|-----------------------|
| `n_estimators`           | [50, 100]             |
| `max_depth`              | [3, 5, 7]             |
| `min_samples_split`      | [10, 15]              |
| `min_samples_leaf`       | [5, 7]                |

### Cross-Validation:

- **Strategy**: 5-fold CV

- **Scoring Metric**: `f1_micro` (suitable for multi-label classification)

---

## Evaluation Metrics

| Metric                  | Description                                        |
|-------------------------|----------------------------------------------------|
| **Accuracy Score**      | Exact match ratio across all labels.               |
| **F1 Score (Micro Avg)**| Harmonic mean of precision and recall (micro avg). |
| **Classification Report**| Detailed report per doctor specialization.         |

- **Training Scores**:

  - F1 Score (micro avg): *reported during training*

- **Test Scores**:

  - Accuracy: *reported during evaluation*

  - F1 Score (micro avg): *reported during evaluation*

---

## Output Artifacts

After training, the following artifacts are saved in the **project root**:

- `trained_appointment_model.pkl`: Trained model file.

- `preprocessing_objects.pkl`: Contains:

  - Scaler

  - TF-IDF vectorizer

  - Label encoders

  - Feature selector

  - Doctor labels

---

## How to Trigger Model Training

Run the **main pipeline** from the project root:

```bash
python model_main.py

```
---
[Back to Main README](../README.md)
