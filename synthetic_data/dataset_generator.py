# File: synthetic_data/dataset_generator.py
# -----------------------------------------

###########################################################################
# A CSV file with around 1000 records using Python (Faker, Pandas, Numpy).
###########################################################################

# Import Libraries 
import pandas as pd
import numpy as np
import random

from faker import Faker

# Import Custon Libraries
from disease_list import diseases

# Initialize Faker
faker = Faker()

# Generate synthetic dataset
number_records = 10000

data = []

for _ in range(number_records):
    patient_id = faker.uuid4()
    name = faker.name()
    age = random.randint(5, 90)
    gender = random.choice(["Male", "Female", "Other"])
    
    # Randomly select a disease and related doctors
    disease = random.choice(list(diseases.keys()))
    doctors_required = diseases[disease]
    
    # Past consultations (random 0-5)
    past_consultations = random.randint(0, 5)
    
    # Appointment details
    appointment_date = faker.date_this_year()
    consultation_duration = random.choice([15, 30, 45, 60])
    doctor_availability = random.choice(["Available", "Busy", "Limited Slots"])
    
    # Symptoms (random placeholder)
    symptoms = faker.sentence(nb_words=6)
    
    data.append([
        patient_id, name, age, gender, disease, past_consultations, doctors_required,
        appointment_date, consultation_duration, doctor_availability, symptoms
    ])
    
    
# Create DataFrame
    
columns = [
    "Patient_ID", "Name", "Age", "Gender", "Existing_Disease", "Past_Consultations", 
    "Doctors_Required", "Appointment_Date", "Consultation_Duration", "Doctor_Availability", "Symptoms"
]

dataframe = pd.DataFrame(data, columns=columns)

# Save to CSV
file_path = "./patient_data.csv"
dataframe.to_csv(file_path, index=False) 