# File: predictor/appointment_predictor.py

###########################################################################
# Appointment Predictor Class                                             #
#                                                                         #
# This module defines a class to predict required doctors for a patient   #
# based on their medical data and suggest an appointment. It uses a       #
# pre-trained machine learning model and preprocessing objects loaded    #
# from saved files.                                                      #
###########################################################################

# Import Libraries
import pandas as pd          # For data manipulation and DataFrame operations
import numpy as np           # For numerical operations (e.g., array handling)
import joblib                # For loading saved model and preprocessing objects
from datetime import datetime  # For generating current date for appointments


class AppointmentPredictor:
    """
    A class to predict required doctors for a patient and suggest appointments.

    This class loads a pre-trained model and preprocessing objects (scaler, TF-IDF,
    label encoders, etc.) from disk, preprocesses patient data, predicts required
    doctors, and generates appointment details.
    """

    def __init__(self, model_path="trained_appointment_model.pkl", preprocessing_path="preprocessing_objects.pkl"):
        """
        Initialize the AppointmentPredictor with model and preprocessing objects.

        Args:
            model_path (str): Path to the saved trained model file (default: "trained_appointment_model.pkl").
            preprocessing_path (str): Path to the saved preprocessing objects file (default: "preprocessing_objects.pkl").
        """
        # Load the trained model (a MultiOutputClassifier with RandomForest)
        self.model = joblib.load(model_path)
        
        # Load preprocessing objects (scaler, TF-IDF, label encoders, etc.) from a dictionary
        preprocessing_objects = joblib.load(preprocessing_path)
        
        # Assign preprocessing objects as instance attributes for use in methods
        self.scaler = preprocessing_objects["scaler"]         # StandardScaler for numeric scaling
        self.tfidf = preprocessing_objects["tfidf"]           # TfidfVectorizer for symptom text
        self.le_dict = preprocessing_objects["le_dict"]       # Dictionary of LabelEncoders for categorical columns
        self.selector = preprocessing_objects["selector"]     # SelectKBest for feature selection
        self.all_doctors = preprocessing_objects["doctors"]   # List of all possible doctor specializations

    def preprocess_patient(self, patient_data):
        """
        Preprocess patient data into a format suitable for model prediction.

        This method transforms raw patient data (e.g., categorical, numeric, text)
        into a numerical feature matrix that matches the model's training format.

        Args:
            patient_data (dict): Dictionary containing patient information (e.g., Age, Symptoms, etc.).

        Returns:
            pd.DataFrame: Preprocessed feature matrix with columns matching the model's expected features.
        """
        # Convert the patient dictionary into a single-row DataFrame
        dataframe = pd.DataFrame([patient_data])

        # Define columns that are not needed for prediction and drop them if present
        cols_to_drop = ["Patient_ID", "Name", "Address", "Contact_Number", "Email", "Preferred_Doctors", "Hospital_Clinic_Name"]
        dataframe = dataframe.drop(columns=[col for col in cols_to_drop if col in dataframe.columns])

        # Convert date columns to days since January 1, 2000, for numerical consistency
        dataframe["Appointment_Date"] = pd.to_datetime(dataframe["Appointment_Date"]).map(
            lambda x: (x - pd.Timestamp("2000-01-01")).days
        )
        dataframe["Diagnosis_Date"] = pd.to_datetime(dataframe["Diagnosis_Date"]).map(
            lambda x: (x - pd.Timestamp("2000-01-01")).days
        )

        # List of categorical columns to encode into numerical values
        categorical_cols = [
            "Gender", "Blood_Group", "Insurance_Provider", "Existing_Disease", "Allergies",
            "Chronic_Conditions", "Previous_Surgeries", "Medication_History", "Family_History",
            "Doctor_Availability", "Consultation_Type", "Disease_Severity_Level", "Lab_Test_Results",
            "Smoking_Alcohol_Use", "Physical_Activity_Level", "Dietary_Preference", "Mental_Health_Conditions"
        ]
        # Encode each categorical column using the saved LabelEncoder
        for col in categorical_cols:
            if col in dataframe.columns:
                le = self.le_dict[col]  # Get the LabelEncoder for this column
                # Handle unseen values by defaulting to the first known class
                dataframe[col] = dataframe[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                dataframe[col] = le.transform(dataframe[col])  # Convert to numeric codes

        # Scale numeric columns to have zero mean and unit variance (same as training)
        numeric_cols = ["Age", "Consultation_Duration_Minutes", "BMI", "Past_Consultations"]
        dataframe[numeric_cols] = self.scaler.transform(dataframe[numeric_cols])

        # Transform the "Symptoms" text into a TF-IDF feature matrix (20 features max)
        symptoms_tfidf = self.tfidf.transform(dataframe["Symptoms"].fillna("")).toarray()
        tfidf_cols = [f"symptom_{i}" for i in range(20)]  # Name columns as symptom_0, symptom_1, etc.
        # Create a DataFrame for TF-IDF features, ensuring the same index as the main DataFrame
        dataframe_tfidf = pd.DataFrame(symptoms_tfidf, columns=tfidf_cols, index=dataframe.index)
        # Replace the original "Symptoms" column with TF-IDF features
        dataframe = pd.concat([dataframe.drop(columns=["Symptoms"]), dataframe_tfidf], axis=1)

        # Get the exact feature names the model was trained on (from the first RandomForest estimator)
        expected_features = self.model.estimators_[0].feature_names_in_
        # Create a feature matrix with only the expected features, initialized to zero
        X = pd.DataFrame(0, index=dataframe.index, columns=expected_features)
        # Copy values from the preprocessed DataFrame for matching features
        for col in expected_features:
            if col in dataframe.columns:
                X[col] = dataframe[col].values  # Use .values to avoid index issues

        return X

    def predict_doctors(self, patient_data):
        """
        Predict the required doctors for a patient based on their preprocessed data.

        Args:
            patient_data (dict): Dictionary containing patient information.

        Returns:
            list: List of doctor specializations (e.g., ["Endocrinologist"]) predicted for the patient.
        """
        # Preprocess the patient data into a feature matrix
        X = self.preprocess_patient(patient_data)
        # Predict using the loaded model (returns a binary array for each doctor)
        y_pred = self.model.predict(X)[0]  # Take the first row since it’s a single patient
        # Convert predictions to a list of doctor names where prediction is 1
        required_doctors = [doctor for doctor, pred in zip(self.all_doctors, y_pred) if pred == 1]
        return required_doctors

    def get_appointment(self, patient_data):
        """
        Suggest an appointment based on the predicted required doctors.

        Args:
            patient_data (dict): Dictionary containing patient information.

        Returns:
            dict or None: Dictionary with appointment details if doctors are predicted, None otherwise.
        """
        # Get the list of required doctors
        required_doctors = self.predict_doctors(patient_data)
        # Check if no doctors are predicted
        if not required_doctors:
            print("No doctors required based on the prediction.")
            return None

        # Generate the current date as the appointment date in YYYY-MM-DD format
        appointment_date = datetime.now().strftime("%Y-%m-%d")
        # Create a dictionary with appointment details
        appointment_details = {
            "Patient_Name": patient_data.get("Name", "Unknown"),  # Default to "Unknown" if Name is missing
            "Required_Doctors": required_doctors,                # List of predicted doctors
            "Appointment_Date": appointment_date,                # Current date
            "Consultation_Type": patient_data.get("Consultation_Type", "In-person"),  # Default to In-person
            "Hospital_Clinic_Name": patient_data.get("Hospital_Clinic_Name", "General Hospital")  # Default hospital
        }

        # Print the appointment details in a readable format
        print("\nAppointment Details:")
        for key, value in appointment_details.items():
            print(f"{key}: {value}")
        return appointment_details


