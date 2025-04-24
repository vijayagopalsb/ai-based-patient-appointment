# MediSched AI
## An AI-Based Patient Appointment System

# 🏥 AI-Based Patient Appointment Scheduling

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/github/license/vijayagopalsb/ai-based-patient-appointment)
![Last Commit](https://img.shields.io/github/last-commit/vijayagopalsb/ai-based-patient-appointment)
![Issues](https://img.shields.io/github/issues/vijayagopalsb/ai-based-patient-appointment)
![Pull Requests](https://img.shields.io/github/issues-pr/vijayagopalsb/ai-based-patient-appointment)
![Forks](https://img.shields.io/github/forks/vijayagopalsb/ai-based-patient-appointment?style=social)
![Stars](https://img.shields.io/github/stars/vijayagopalsb/ai-based-patient-appointment?style=social)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![Made with ML](https://img.shields.io/badge/Made%20with-ML-blue)


## Overview

The **AI-Based Patient Appointment System** is a machine learning-driven solution that predicts the required medical specialists based on patient data. The system uses synthetic patient data, preprocesses it, trains a predictive model, and provides recommendations for doctor appointments. 

## Features

- Synthetic Data Generation: Generates realistic patient data for model training.

- Data Preprocessing: Cleans and transforms raw data for machine learning.

- Model Training: Builds and trains a predictive model using scikit-learn.

- Doctor Prediction: Predicts the required doctor based on patient input.

- Logging Mechanism: Tracks system operations and errors.

## Documentation

**Documentation**
- [Setup Guide](docs/setup.md)
- [Exploratory Data Analysis](docs/eda.md)
- [Model Training](docs/training.md)
- [Model Evaluation Metrics](docs/evaluation.md)


## Project Structure

<pre>

ai-based-patient-appointment/
├── data/
│   └── synthetic_patient_data.csv
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── eda/
│   │   └── eda_analysis.py
│   ├── generator/
│   │   └── patient_data_generator.py
│   ├── preprocessor/
│   │   └── data_preprocessor.py
│   ├── trainer/
│   │   └── model_trainer.py
│   └── utils/
│       ├── config.py
│       └── logging_config.py
├── model_main.py
├── requirements.txt
└── README.md

</pre>

## Installation & Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.10+

- Required Python libraries:

```python
pip install pandas numpy scikit-learn joblib faker
```
## Running the System

1. Generate Patient Data

```python
python data_generator/patient_data_generator.py
```

2. Preprocess Data

```python
python data_preprocessor/data_preprocessor.py
```

3. Train the Model

```python
python model_trainer/model_trainer.py
```

4. Run Model

```python
python model_main.py
```

4. Run Predictions

```python
python client_main.py
```

## Logging
The system logs critical information and errors using `logging_config.py`. Logs are stored in the `logs/` directory.

## Output

<pre>
2025-04-23 14:53:01,097 - INFO -->> Starting Project ...
2025-04-23 14:53:01,113 - INFO -->> Generating "15000" Synthetic Patient Data ...
2025-04-23 14:53:07,810 - INFO -->> Dataset with 15000 records generated and saved to data/synthetic_patient_data.csv!
2025-04-23 14:53:08,451 - INFO -->> Starting Visual EDA Analysis...
2025-04-23 14:53:08,842 - INFO -->> Saved Age-Distribution Plot to output_images/eda directory
2025-04-23 14:53:08,974 - INFO -->> Saved Gender-Distribution Plot to output_images/eda directory
2025-04-23 14:53:09,180 - INFO -->> Saved Disease-Distribution Plot to output_images/eda directory
2025-04-23 14:53:09,313 - INFO -->> Saved Heatmap Plot to output_images/eda directory
2025-04-23 14:53:09,431 - INFO -->> Saved Dr_workload Plot to output_images/eda directory
2025-04-23 14:53:09,473 - INFO -->> Stoped Visual EDA Analysis Successfully.
2025-04-23 14:53:09,473 - INFO -->> Splitting data into training and testing sets...
2025-04-23 14:53:09,481 - INFO -->> Data split completed.
2025-04-23 14:53:09,481 - INFO -->> Starting GridSearchCV for hyperparameter tuning...
2025-04-23 14:53:09,481 - INFO -->> Starting hyperparameter tuning using GridSearchCV
2025-04-23 14:53:52,931 - INFO -->> GridSearchCV Completed. Best Parameters: {'estimator__max_depth': 7, 'estimator__min_samples_leaf': 5, 'estimator__min_samples_split': 10, 'estimator__n_estimators': 100}
2025-04-23 14:53:52,931 - INFO -->> Best Parameters: {'estimator__max_depth': 7, 'estimator__min_samples_leaf': 5, 'estimator__min_samples_split': 10, 'estimator__n_estimators': 100}
2025-04-23 14:53:52,931 - INFO -->> Grid search completed.
2025-04-23 14:53:52,931 - INFO -->> Performing cross-validation...
2025-04-23 14:54:09,066 - INFO -->> Cross-Validation F1 Score: 0.92 (�0.02)
2025-04-23 14:54:09,066 - INFO -->> Cross-validation completed.
2025-04-23 14:54:09,066 - INFO -->> Evaluating training and test performance...
2025-04-23 14:54:09,378 - INFO -->> Training F1 Score: 0.96
2025-04-23 14:54:09,478 - INFO -->> Test Accuracy: 0.91
2025-04-23 14:54:09,478 - INFO -->> Test F1 Score: 0.96
2025-04-23 14:54:09,478 - INFO -->> Test Classification Report:
2025-04-23 14:54:09,478 - INFO -->> 
                    precision    recall  f1-score   support

      Cardiologist       1.00      0.87      0.93       862
     Dermatologist       1.00      1.00      1.00       262
   Endocrinologist       0.96      1.00      0.98       397
Gastroenterologist       1.00      1.00      1.00       227
       Neurologist       0.94      1.00      0.97       292
      Psychiatrist       0.99      1.00      0.99       227
     Pulmonologist       0.87      0.93      0.90       588
    Rheumatologist       1.00      1.00      1.00       260

         micro avg       0.96      0.95      0.96      3115
         macro avg       0.97      0.97      0.97      3115
      weighted avg       0.96      0.95      0.96      3115
       samples avg       0.97      0.97      0.96      3115

2025-04-23 14:54:09,597 - INFO -->> Evaluation completed. Saving model...
2025-04-23 14:54:09,597 - INFO -->> Model and preprocessing objects saved to 'models' directory!
2025-04-23 14:54:09,597 - INFO -->> Successfully Completed Model Training and Testing.

2025-04-23 14:54:26,956 - INFO -->> Starting patient appointment demo...
2025-04-23 14:54:26,956 - INFO -->> Scheduled appointments for patients across various age groups.


2025-04-23 14:58:21,157 - INFO -->> MAKING APPOINMENT DEMO - Client Part


2025-04-23 14:58:21,157 - INFO -->> Starting patient appointment demo...
2025-04-23 14:58:21,157 - INFO -->> Scheduled appointments for patients across various age groups.
2025-04-23 14:58:21,912 - INFO -->> 

Predicted Doctors: ['Cardiologist']
2025-04-23 14:58:21,954 - INFO -->> Appointment Details:
2025-04-23 14:58:21,955 - INFO -->> Patient_Name: Mary Johnson
2025-04-23 14:58:21,955 - INFO -->> Required_Doctors: ['Cardiologist']
2025-04-23 14:58:21,955 - INFO -->> Appointment_Date: 2025-04-23
2025-04-23 14:58:21,955 - INFO -->> Consultation_Type: In-person
2025-04-23 14:58:21,955 - INFO -->> Hospital_Clinic_Name: City Hospital
2025-04-23 14:58:21,994 - INFO -->> 

Predicted Doctors: ['Pulmonologist']
2025-04-23 14:58:22,030 - INFO -->> Appointment Details:
2025-04-23 14:58:22,030 - INFO -->> Patient_Name: Robert Patel
2025-04-23 14:58:22,030 - INFO -->> Required_Doctors: ['Pulmonologist']
2025-04-23 14:58:22,030 - INFO -->> Appointment_Date: 2025-04-23
2025-04-23 14:58:22,030 - INFO -->> Consultation_Type: Online
2025-04-23 14:58:22,030 - INFO -->> Hospital_Clinic_Name: Health Clinic
2025-04-23 14:58:22,076 - INFO -->> 

Predicted Doctors: ['Neurologist']
2025-04-23 14:58:22,114 - INFO -->> Appointment Details:
2025-04-23 14:58:22,114 - INFO -->> Patient_Name: Sarah Kim
2025-04-23 14:58:22,114 - INFO -->> Required_Doctors: ['Neurologist']
2025-04-23 14:58:22,114 - INFO -->> Appointment_Date: 2025-04-23
2025-04-23 14:58:22,114 - INFO -->> Consultation_Type: Online
2025-04-23 14:58:22,114 - INFO -->> Hospital_Clinic_Name: Wellness Center
2025-04-23 14:58:22,114 - INFO -->> Successfully completed patient appointment demo ...

</pre>


### Future Enhancements

- Integrate XGBoost or LightGBM for better performance.

- Experiment with multi-label neural networks (Keras/PyTorch).

- Add model interpretability using SHAP or LIME.


## Contributors

- Vijayagopal S - [GitHub](https://github.com/vijayagopalsb)

## License

This project is licensed under the Apache License 2.0.
