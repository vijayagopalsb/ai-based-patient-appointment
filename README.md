# MediSched AI
## An AI-Based Patient Appointment System

## Overview

The **AI-Based Patient Appointment System** is a machine learning-driven solution that predicts the required medical specialists based on patient data. The system uses synthetic patient data, preprocesses it, trains a predictive model, and provides recommendations for doctor appointments. 

## Features

- Synthetic Data Generation: Generates realistic patient data for model training.

- Data Preprocessing: Cleans and transforms raw data for machine learning.

- Model Training: Builds and trains a predictive model using scikit-learn.

- Doctor Prediction: Predicts the required doctor based on patient input.

- Logging Mechanism: Tracks system operations and errors.

## Project Structure

<pre>
AI-Based-Patient-Appointment/
│── data_generator/
│   └── patient_data_generator.py  # Generates synthetic patient data
│── data_preprocessor/
│   └── data_preprocessor.py       # Preprocesses raw patient data
│── model_trainer/
│   └── model_trainer.py           # Trains machine learning model
│── predictor/
│   └── appointment_predictor.py   # Predicts doctor based on patient input
│── model_main.py                  # Main script to train and save model
│── client_main.py                  # Client interface for making predictions
│── logging_config.py               # Logging configuration for system events
│── preprocessing_objects.pkl       # Saved preprocessing objects (scaler, encoders, etc.)
│── synthetic_patient_data.csv      # Generated patient dataset
│── trained_appointment_model.pkl   # Trained ML model
│── README.md                      # Project documentation
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
2025-03-27 19:38:21,768 - INFO ->>> --------------------------------------------------------------------------------
2025-03-27 19:38:21,768 - INFO ->>> Starting Project ...
2025-03-27 19:38:21,768 - INFO ->>> Generating "10000" Synthetic Patient Data ...
2025-03-27 19:38:26,037 - INFO ->>> Dataset with 10000 records generated and saved to synthetic_patient_data.csv!
2025-03-27 19:38:51,908 - INFO ->>> Best Parameters: {'estimator__max_depth': 7, 'estimator__min_samples_leaf': 7, 'estimator__min_samples_split': 10, 'estimator__n_estimators': 100}
2025-03-27 19:39:02,433 - INFO ->>> Cross-Validation F1 Score: 0.95 (�0.01)
2025-03-27 19:39:02,643 - INFO ->>> Training F1 Score: 0.95
2025-03-27 19:39:02,741 - INFO ->>> Test Accuracy: 0.88
2025-03-27 19:39:02,742 - INFO ->>> Test F1 Score: 0.94
2025-03-27 19:39:02,742 - INFO ->>> Test Classification Report:
2025-03-27 19:39:02,743 - INFO ->>>                     precision    recall  f1-score   support

      Cardiologist       0.92      0.88      0.90       544
     Dermatologist       1.00      1.00      1.00       161
   Endocrinologist       0.94      1.00      0.97       251
Gastroenterologist       0.98      0.98      0.98       207
       Neurologist       0.78      1.00      0.88       204
      Psychiatrist       0.88      1.00      0.94       145
     Pulmonologist       0.91      1.00      0.96       393
    Rheumatologist       1.00      1.00      1.00       173

         micro avg       0.92      0.97      0.94      2078
         macro avg       0.93      0.98      0.95      2078
      weighted avg       0.92      0.97      0.94      2078
       samples avg       0.95      0.98      0.96      2078

2025-03-27 19:39:02,922 - INFO ->>> Model and preprocessing objects saved!
2025-03-27 19:39:02,922 - INFO ->>> Successfully Completed Model Training and Testing...

2025-03-27 19:39:11,958 - INFO ->>> Starting patient appointment demo...
2025-03-27 19:39:12,881 - INFO ->>> Predicted Doctors: ['Cardiologist']
2025-03-27 19:39:12,918 - INFO ->>> Appointment Details:
2025-03-27 19:39:12,918 - INFO ->>> Patient_Name: Mary Johnson
2025-03-27 19:39:12,918 - INFO ->>> Required_Doctors: ['Cardiologist']
2025-03-27 19:39:12,918 - INFO ->>> Appointment_Date: 2025-03-27
2025-03-27 19:39:12,918 - INFO ->>> Consultation_Type: In-person
2025-03-27 19:39:12,918 - INFO ->>> Hospital_Clinic_Name: City Hospital
2025-03-27 19:39:12,951 - INFO ->>> Predicted Doctors: ['Pulmonologist']
2025-03-27 19:39:12,985 - INFO ->>> Appointment Details:
2025-03-27 19:39:12,985 - INFO ->>> Patient_Name: Robert Patel
2025-03-27 19:39:12,985 - INFO ->>> Required_Doctors: ['Pulmonologist']
2025-03-27 19:39:12,985 - INFO ->>> Appointment_Date: 2025-03-27
2025-03-27 19:39:12,985 - INFO ->>> Consultation_Type: Online
2025-03-27 19:39:12,985 - INFO ->>> Hospital_Clinic_Name: Health Clinic
2025-03-27 19:39:13,034 - INFO ->>> Predicted Doctors: ['Neurologist']
2025-03-27 19:39:13,068 - INFO ->>> Appointment Details:
2025-03-27 19:39:13,068 - INFO ->>> Patient_Name: Sarah Kim
2025-03-27 19:39:13,068 - INFO ->>> Required_Doctors: ['Neurologist']
2025-03-27 19:39:13,068 - INFO ->>> Appointment_Date: 2025-03-27
2025-03-27 19:39:13,068 - INFO ->>> Consultation_Type: Online
2025-03-27 19:39:13,068 - INFO ->>> Hospital_Clinic_Name: Wellness Center
2025-03-27 19:39:13,068 - INFO ->>> Successfully completed patient appointment demo ...

</pre>

## Documentation

📚 **Documentation**
- [Setup Guide](docs/setup.md)
- [Model Training](docs/training.md)


## Contributors

- Vijayagopal S - [GitHub](https://github.com/vijayagopalsb)

## License

This project is licensed under the Apache License 2.0.
