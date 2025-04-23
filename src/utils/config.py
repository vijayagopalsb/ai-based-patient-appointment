# config.py

class Config:

    # Logging settings
    LOG_FILE = "logs/ai_patient_appointment.log"

    # Logging level
    LOG_LEVEL = "INFO"

    # Numbner of synthetic data 
    NUMBER_OF_PATIENT_RECORDS = 15000

    # File name
    SYNTHETIC_DATA = "data/synthetic_patient_data.csv"

    # Appoinment model path and name
    MODEL_PATH_NAME="models/trained_appointment_model.pkl"

    PREPROCESSING_OBJECTS_PATH = "models/preprocessing_objects.pkl"
