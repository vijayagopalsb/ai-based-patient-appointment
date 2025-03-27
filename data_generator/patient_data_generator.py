# File: patient_data_generator.py

###########################################################################
# Patient Data Generator Class                                            #
#                                                                         #
# This module defines a class to generate synthetic patient data for      #
# training or testing machine learning models in healthcare applications. #
###########################################################################

# Import Libraries
import pandas as pd
import random
from faker import Faker

# Import Logger
from logging_config import logger

class PatientDataGenerator:
    """A class to generate synthetic patient data with realistic attributes."""

    def __init__(self):
        """Initialize the PatientDataGenerator with Faker and predefined categories."""
        # Initialize Faker for generating realistic synthetic data
        self.faker = Faker()

        # Define disease-to-specialization mapping
        self.diseases = {
            "Hypertension": ["Cardiologist"],
            "Diabetes": ["Endocrinologist"],
            "Asthma": ["Pulmonologist"],
            "Migraine": ["Neurologist"],
            "Arthritis": ["Rheumatologist"],
            "Heart Disease": ["Cardiologist"],
            "Pneumonia": ["Pulmonologist"],
            "Depression": ["Psychiatrist"],
            "Skin Allergy": ["Dermatologist"],
            "Gastroenteritis": ["Gastroenterologist"]
        }

        # Define categorical options for patient attributes
        self.blood_groups = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
        self.insurance_providers = ["None", "MediCare", "HealthFirst", "Aetna", "Blue Cross"]
        self.consultation_types = ["Online", "In-person"]
        self.severity_levels = ["Mild", "Moderate", "Severe"]
        self.activity_levels = ["Sedentary", "Active", "Highly Active"]
        self.dietary_preferences = ["Vegetarian", "Non-Vegetarian", "Vegan"]

    def generate_synthetic_data(self, n_records=10000, output_file="synthetic_patient_data.csv"):
        """
        Generate synthetic patient data and save it to a CSV file.

        Args:
            n_records (int): Number of patient records to generate (default: 10000).
            output_file (str): Path to save the generated CSV file (default: "synthetic_patient_data.csv").

        Returns:
            pd.DataFrame: DataFrame containing the generated synthetic patient data.
        """
        data = []  # List to store patient records

        # Generate each patient record
        for _ in range(n_records):
            # Generate age with a range from 5 to 90
            age = random.randint(5, 90)

            # Introduce age-disease correlation: older patients more likely to have certain diseases
            if age > 60:
                disease = random.choices(
                    list(self.diseases.keys()), 
                    weights=[0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
                )[0]
            else:
                disease = random.choice(list(self.diseases.keys()))

            # Assign doctor specializations based on disease
            specializations = self.diseases[disease]
            # Multi-label scenario: 30% chance Diabetes requires a Cardiologist too
            if disease == "Diabetes" and random.random() < 0.3:
                specializations = ["Endocrinologist", "Cardiologist"]

            # Generate patient attributes using Faker and random choices
            patient_id = self.faker.uuid4()  # Unique patient identifier
            name = self.faker.name()         # Random realistic name
            gender = random.choice(["Male", "Female", "Other"])  # Gender selection
            blood_group = random.choice(self.blood_groups)       # Blood group from predefined list
            address = self.faker.address()   # Random address
            contact_number = self.faker.phone_number()  # Random phone number
            email = self.faker.email()       # Random email address
            insurance_provider = random.choice(self.insurance_providers)  # Insurance selection
            allergies = random.choice(["None", "Peanuts", "Pollen", "Dust", "Penicillin"])  # Allergy options
            chronic_conditions = random.choice(["None", "Hypertension", "Diabetes", "Asthma"])  # Chronic conditions
            previous_surgeries = random.choice(["None", "Appendectomy", "Knee Replacement", "Gallbladder Removal"])  # Surgery history
            medication_history = random.choice(["None", "Metformin", "Aspirin", "Ibuprofen"])  # Medication history
            family_history = random.choice(["None", "Heart Disease", "Diabetes", "Cancer"])  # Family medical history
            past_consultations = random.randint(0, 20)  # Number of past consultations
            preferred_doctor = self.faker.name()  # Random doctor name
            appointment_date = self.faker.date_this_year()  # Random date in the current year
            consultation_duration_minutes = random.choice([15, 30, 45, 60])  # Duration in minutes
            doctor_availability = random.choice(["Available", "Busy", "Limited Slots"])  # Doctor availability status
            consultation_type = random.choice(self.consultation_types)  # Online or In-person
            hospital_clinic_name = self.faker.company()  # Random hospital/clinic name
            symptoms = self.faker.sentence(nb_words=6)  # Random 6-word symptom description
            disease_severity_level = random.choice(self.severity_levels)  # Severity level
            diagnosis_date = self.faker.date_this_decade()  # Random date in the last decade
            lab_test_results = random.choice(["Normal", "Abnormal", "Borderline"])  # Lab test outcome
            smoking_alcohol_use = random.choice(["Yes", "No"])  # Lifestyle factor
            bmi = round(random.uniform(18.5, 35.0), 1)  # BMI between 18.5 and 35.0
            physical_activity_level = random.choice(self.activity_levels)  # Activity level
            dietary_preference = random.choice(self.dietary_preferences)  # Dietary preference
            mental_health_conditions = random.choice(["None", "Anxiety", "Depression", "Bipolar Disorder"])  # Mental health status

            # Append the patient record as a list
            data.append([
                patient_id, name, age, gender, blood_group, address, contact_number, email,
                insurance_provider, disease, allergies, chronic_conditions, previous_surgeries,
                medication_history, family_history, past_consultations, specializations, preferred_doctor,
                appointment_date, consultation_duration_minutes, doctor_availability, consultation_type,
                hospital_clinic_name, symptoms, disease_severity_level, diagnosis_date,
                lab_test_results, smoking_alcohol_use, bmi, physical_activity_level, dietary_preference,
                mental_health_conditions
            ])

        # Define column names for the DataFrame
        columns = [
            "Patient_ID", "Name", "Age", "Gender", "Blood_Group", "Address", "Contact_Number", "Email",
            "Insurance_Provider", "Existing_Disease", "Allergies", "Chronic_Conditions", "Previous_Surgeries",
            "Medication_History", "Family_History", "Past_Consultations", "Doctors_Required", "Preferred_Doctors",
            "Appointment_Date", "Consultation_Duration_Minutes", "Doctor_Availability", "Consultation_Type",
            "Hospital_Clinic_Name", "Symptoms", "Disease_Severity_Level", "Diagnosis_Date",
            "Lab_Test_Results", "Smoking_Alcohol_Use", "BMI", "Physical_Activity_Level", "Dietary_Preference",
            "Mental_Health_Conditions"
        ]

        # Create a DataFrame from the generated data
        df = pd.DataFrame(data, columns=columns)

        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Dataset with {n_records} records generated and saved to {output_file}!")

        return df

# Example usage
if __name__ == "__main__":
    generator = PatientDataGenerator()
    df = generator.generate_synthetic_data(n_records=100, output_file="synthetic_patient_data_small.csv")
    logger.info(df.head())  # Display the first 5 rows for verification