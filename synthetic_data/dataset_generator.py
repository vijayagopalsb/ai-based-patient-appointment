# File: synthetic_data/dataset_generator.py
# -----------------------------------------

###########################################################################
# Generates a synthetic dataset of patient records (10,000 records)      #
# The dataset includes patient details, medical history, appointments,   #
# and doctor consultation requirements.                                  #
# The generated dataset is stored as a CSV file.                         #
###########################################################################

# Import Libraries 
import pandas as pd
import numpy as np
import random

from faker import Faker

# Import Custom Libraries
from disease_list import (
    diseases, blood_groups, insurance_providers, 
    consultation_types, severity_levels, activity_levels, dietary_preferences
)

# Initialize Faker instance for generating fake data
faker = Faker()

# Define the number of records to generate
number_records = 500

# Create an empty list to store patient records
data = []

for _ in range(number_records):
    # Select a random disease and its corresponding specializations
    disease = random.choice(list(diseases.keys()))
    specializations = diseases[disease]  # Now supports multiple doctors per disease
    
    # Generate patient details
    patient_id = faker.uuid4()
    name = faker.name()
    age = random.randint(5, 90)
    gender = random.choice(["Male", "Female", "Other"])
    blood_group = random.choice(blood_groups)
    address = faker.address()
    contact_number = faker.phone_number()
    email = faker.email()
    insurance_provider = random.choice(insurance_providers)
    existing_disease = disease
    
    # Medical history
    allergies = random.choice(["None", "Peanuts", "Pollen", "Dust", "Penicillin"])
    chronic_conditions = random.choice(["None", "Hypertension", "Diabetes", "Asthma"])
    previous_surgeries = random.choice(["None", "Appendectomy", "Knee Replacement", "Gallbladder Removal"])
    medication_history = random.choice(["None", "Metformin", "Aspirin", "Ibuprofen"])
    family_history = random.choice(["None", "Heart Disease", "Diabetes", "Cancer"])
    past_consultations = random.randint(0, 20)
    preferred_doctor = faker.name()
    
    # Appointment details
    appointment_date = faker.date_this_year()
    consultation_duration_minutes = random.choice([15, 30, 45, 60])
    doctor_availability = random.choice(["Available", "Busy", "Limited Slots"])
    consultation_type = random.choice(consultation_types)
    hospital_clinic_name = faker.company()
    
    # Disease-related information
    symptoms = faker.sentence(nb_words=6)
    disease_severity_level = random.choice(severity_levels)
    diagnosis_date = faker.date_this_decade()
    lab_test_results = random.choice(["Normal", "Abnormal", "Borderline"])
    
    # Additional patient details
    smoking_alcohol_use = random.choice(["Yes", "No"])
    bmi = round(random.uniform(18.5, 35.0), 1)
    physical_activity_level = random.choice(activity_levels)
    dietary_preference = random.choice(dietary_preferences)
    mental_health_conditions = random.choice(["None", "Anxiety", "Depression", "Bipolar Disorder"])
    
    # Append generated data to the list
    data.append([
        patient_id, name, age, gender, blood_group, address, contact_number, email, 
        insurance_provider, existing_disease, allergies, chronic_conditions, previous_surgeries, 
        medication_history, family_history, past_consultations, specializations, preferred_doctor, 
        appointment_date, consultation_duration_minutes, doctor_availability, consultation_type, 
        hospital_clinic_name, symptoms, disease_severity_level, diagnosis_date, 
        lab_test_results, smoking_alcohol_use, bmi, physical_activity_level, dietary_preference, 
        mental_health_conditions
    ])
    
# Define column names for the dataset
columns = [
    "Patient_ID", "Name", "Age", "Gender", "Blood_Group", "Address", "Contact_Number", "Email", 
    "Insurance_Provider", "Existing_Disease", "Allergies", "Chronic_Conditions", "Previous_Surgeries", 
    "Medication_History", "Family_History", "Past_Consultations", "Doctors_Required", "Preferred_Doctors", 
    "Appointment_Date", "Consultation_Duration_Minutes", "Doctor_Availability", "Consultation_Type", 
    "Hospital_Clinic_Name", "Symptoms", "Disease_Severity_Level", 
    "Diagnosis_Date", "Lab_Test_Results", "Smoking_Alcohol_Use", "BMI", "Physical_Activity_Level", 
    "Dietary_Preference", "Mental_Health_Conditions"
]

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=columns)

# Save dataset to a CSV file
df.to_csv("synthetic_patient_data.csv", index=False, encoding="utf-8")

print("Dataset generated and saved successfully!")