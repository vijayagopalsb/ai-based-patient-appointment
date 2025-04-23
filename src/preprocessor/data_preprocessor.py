# File: data_preprocessor/data_preprocessor.py

#
# Data Preprocessor Class                                                 
#                                                                         
# This module defines a class to preprocess synthetic patient data for    
# machine learning model training. It handles feature encoding, scaling,  
# TF-IDF transformation, and multi-label target preparation.             
#

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

# Custom App Libraries
from src.utils.logging_config import logger

class DataPreprocessor:
    """
    A class to preprocess patient data for machine learning model training.
    """

    def __init__(self):
        """Initialize the DataPreprocessor with no initial state."""
        pass  # No initialization needed; all preprocessing objects are created in the method

    def preprocess_data(self, df):
        """
        Preprocess patient data into features and multi-label targets.
        """
        # --- Separate Target and Features ---
        # Extract the "Doctors_Required" column (list of doctors) as the target
        y_temp = df["Doctors_Required"]
        # Create a feature DataFrame by dropping the target column
        df_features = df.drop(columns=["Doctors_Required"])

        # --- Clean Features ---
        # Define columns that are not relevant for prediction and drop them
        cols_to_drop = ["Patient_ID", "Name", "Address", "Contact_Number", "Email", "Preferred_Doctors", "Hospital_Clinic_Name"]
        df_features = df_features.drop(columns=cols_to_drop)

        # Remove duplicate rows based on features (preserves unique feature combinations)
        df_features = df_features.drop_duplicates()

        # Reattach the target column, aligning with the cleaned feature indices
        df = df_features.join(y_temp.loc[df_features.index])

        # --- Date Conversion ---
        # Convert date columns to numeric days since January 1, 2000, for consistency
        df["Appointment_Date"] = pd.to_datetime(df["Appointment_Date"]).map(
            lambda x: (x - pd.Timestamp("2000-01-01")).days
        )
        df["Diagnosis_Date"] = pd.to_datetime(df["Diagnosis_Date"]).map(
            lambda x: (x - pd.Timestamp("2000-01-01")).days
        )

        # --- Categorical Encoding ---
        # List of categorical columns to encode into numeric values
        categorical_cols = [
            "Gender", "Blood_Group", "Insurance_Provider", "Existing_Disease", "Allergies",
            "Chronic_Conditions", "Previous_Surgeries", "Medication_History", "Family_History",
            "Doctor_Availability", "Consultation_Type", "Disease_Severity_Level", "Lab_Test_Results",
            "Smoking_Alcohol_Use", "Physical_Activity_Level", "Dietary_Preference", "Mental_Health_Conditions"
        ]
        # Dictionary to store fitted LabelEncoders for each categorical column
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()  # Create a new LabelEncoder instance
            df[col] = le.fit_transform(df[col])  # Fit and transform the column to numeric codes
            le_dict[col] = le  # Save the encoder for later use (e.g., prediction)

        # --- Numeric Scaling ---
        # List of numeric columns to scale to zero mean and unit variance
        numeric_cols = ["Age", "Consultation_Duration_Minutes", "BMI", "Past_Consultations"]
        scaler = StandardScaler()  # Create a StandardScaler instance
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])  # Scale numeric columns

        # --- TF-IDF Transformation for Symptoms ---
        # Initialize TF-IDF vectorizer with a maximum of 20 features
        tfidf = TfidfVectorizer(max_features=20)
        # Transform "Symptoms" text into a sparse TF-IDF matrix and convert to dense array
        symptoms_tfidf = tfidf.fit_transform(df["Symptoms"].fillna("")).toarray()
        # Name TF-IDF columns (e.g., symptom_0, symptom_1, etc.)
        tfidf_cols = [f"symptom_{i}" for i in range(20)]
        # Create a DataFrame for TF-IDF features, aligning indices with the main DataFrame
        df_tfidf = pd.DataFrame(symptoms_tfidf, columns=tfidf_cols, index=df.index)
        # Replace the "Symptoms" column with TF-IDF features
        df = pd.concat([df.drop(columns=["Symptoms"]), df_tfidf], axis=1)

        # --- Prepare Multi-Label Target ---
        # Get a sorted list of all unique doctor specializations from the target column
        all_doctors = sorted(set([doc for sublist in df["Doctors_Required"] for doc in sublist]))
        # Create a binary target matrix (rows: patients, columns: doctors)
        y = pd.DataFrame(0, index=df.index, columns=all_doctors)
        # Fill the target matrix: 1 if a doctor is required for a patient, 0 otherwise
        for idx, doctors in enumerate(df["Doctors_Required"]):
            for doc in doctors:
                y.loc[idx, doc] = 1

        # --- Feature Selection ---
        # Separate features (X) from the target column
        X = df.drop(columns=["Doctors_Required"])
        # Use SelectKBest with f_classif to select the top 30 features
        selector = SelectKBest(f_classif, k=30)
        # Fit and transform features, using a proxy target (most frequent doctor) for multi-label
        X_selected = selector.fit_transform(X, y.idxmax(axis=1))
        # Get the names of the selected features
        selected_features = X.columns[selector.get_support()].tolist()

        # Create a DataFrame with the selected features
        X = pd.DataFrame(X_selected, columns=selected_features, index=df.index)

        # Return preprocessed data and preprocessing objects
        return X, y, scaler, tfidf, le_dict, selector, all_doctors


# Example usage
if __name__ == "__main__":
    # Load synthetic data for testing
    df = pd.read_csv("synthetic_patient_data.csv")
    
    # Create an instance of the preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess the data
    X, y, scaler, tfidf, le_dict, selector, all_doctors = preprocessor.preprocess_data(df)
    
    # Print some results for verification
    logger.info("Preprocessed Features Shape:", X.shape)
    logger.info("Selected Features:", X.columns.tolist())
    logger.info("Target Shape:", y.shape)
    logger.info("Unique Doctors:", all_doctors)