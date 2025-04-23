# File: client_main.py

# Import App Libraries
from src.predictor.appointment_predictor import AppointmentPredictor
from src.utils.logging_config import logger
from src.utils.config import Config

# Example usage (can be removed in production)
if __name__ == "__main__":
    
    # List of patients
    patients = [
        # Patient 1: Elderly Female with Heart Disease
        {
            "Patient_ID": "67890", 
            "Name": "Mary Johnson", 
            "Age": 72, 
            "Gender": "Female", 
            "Blood_Group": "O+", 
            "Address": "456 Oak Ave", 
            "Contact_Number": "555-5678", 
            "Email": "mary.johnson@example.com", 
            "Insurance_Provider": "Aetna", 
            "Existing_Disease": "Heart Disease", 
            "Allergies": "Pollen", 
            "Chronic_Conditions": "Hypertension", 
            "Previous_Surgeries": "Bypass Surgery", 
            "Medication_History": "Aspirin", 
            "Family_History": "Heart Disease", 
            "Past_Consultations": 12, 
            "Preferred_Doctors": "Dr. Lee", 
            "Appointment_Date": "2025-04-01", 
            "Consultation_Duration_Minutes": 45, 
            "Doctor_Availability": "Available", 
            "Consultation_Type": "In-person", 
            "Hospital_Clinic_Name": "City Hospital", 
            "Symptoms": "Chest pain and shortness of breath", 
            "Disease_Severity_Level": "Severe", 
            "Diagnosis_Date": "2020-06-10", 
            "Lab_Test_Results": "Abnormal", 
            "Smoking_Alcohol_Use": "No", 
            "BMI": 30.2, 
            "Physical_Activity_Level": "Sedentary", 
            "Dietary_Preference": "Non-Vegetarian", 
            "Mental_Health_Conditions": "Anxiety"
        },
        # Patient 2: Middle-Aged Male with Asthma
        {
            "Patient_ID": "54321", 
            "Name": "Robert Patel", 
            "Age": 45, 
            "Gender": "Male", 
            "Blood_Group": "B-", 
            "Address": "789 Pine Rd", 
            "Contact_Number": "555-9012", 
            "Email": "robert.patel@example.com", 
            "Insurance_Provider": "Blue Cross", 
            "Existing_Disease": "Asthma", 
            "Allergies": "Dust", 
            "Chronic_Conditions": "None", 
            "Previous_Surgeries": "None", 
            "Medication_History": "Albuterol", 
            "Family_History": "Asthma", 
            "Past_Consultations": 3, 
            "Preferred_Doctors": "Dr. Garcia", 
            "Appointment_Date": "2025-03-30", 
            "Consultation_Duration_Minutes": 30, 
            "Doctor_Availability": "Limited Slots", 
            "Consultation_Type": "Online", 
            "Hospital_Clinic_Name": "Health Clinic", 
            "Symptoms": "Wheezing and difficulty breathing", 
            "Disease_Severity_Level": "Moderate", 
            "Diagnosis_Date": "2022-11-05", 
            "Lab_Test_Results": "Borderline", 
            "Smoking_Alcohol_Use": "Yes", 
            "BMI": 26.8, 
            "Physical_Activity_Level": "Active", 
            "Dietary_Preference": "Vegan", 
            "Mental_Health_Conditions": "None"
        },
        # Patient 3: Young Female with Migraine
        {
            "Patient_ID": "98765", 
            "Name": "Sarah Kim", 
            "Age": 28, 
            "Gender": "Female", 
            "Blood_Group": "AB+", 
            "Address": "321 Elm St", 
            "Contact_Number": "555-3456", 
            "Email": "sarah.kim@example.com", 
            "Insurance_Provider": "HealthFirst", 
            "Existing_Disease": "Migraine", 
            "Allergies": "None", 
            "Chronic_Conditions": "None", 
            "Previous_Surgeries": "None", 
            "Medication_History": "Ibuprofen", 
            "Family_History": "Migraine", 
            "Past_Consultations": 2, 
            "Preferred_Doctors": "Dr. Brown", 
            "Appointment_Date": "2025-04-05", 
            "Consultation_Duration_Minutes": 15, 
            "Doctor_Availability": "Available", 
            "Consultation_Type": "Online", 
            "Hospital_Clinic_Name": "Wellness Center", 
            "Symptoms": "Throbbing headache and nausea", 
            "Disease_Severity_Level": "Mild", 
            "Diagnosis_Date": "2024-02-20", 
            "Lab_Test_Results": "Normal", 
            "Smoking_Alcohol_Use": "No", 
            "BMI": 22.5, 
            "Physical_Activity_Level": "Highly Active", 
            "Dietary_Preference": "Vegetarian", 
            "Mental_Health_Conditions": "None"
        }
    ]
    
    logger.info(f"Starting patient appointment demo...")
    
    # Create an instance of the predictor and test it
    predictor = AppointmentPredictor(model_path=Config.MODEL_PATH_NAME,preprocessing_path=Config.PREPROCESSING_OBJECTS_PATH)
    
    # Example usage: Print each patient's name and disease
    for patient in patients:
        doctors = predictor.predict_doctors(patient)
        
        logger.info(f"\n\nPredicted Doctors: {doctors}")
        predictor.get_appointment(patient)
     
    logger.info(f"Successfully completed patient appointment demo ...")   
        
    