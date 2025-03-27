import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained model and preprocessing objects
model = joblib.load("trained_appointment_model.pkl")
preprocessing_objects = joblib.load("preprocessing_objects.pkl")

# Extract preprocessing tools
scaler = preprocessing_objects["scaler"]
tfidf = preprocessing_objects["tfidf"]
train_features = preprocessing_objects["selector"].get_feature_names_out()

# Define patient data (Example)
patient_data = pd.DataFrame({
    "Age": [45],
    "Consultation_Duration_Minutes": [30],
    "BMI": [24.5],
    "Past_Consultations": [3],
    "Symptoms": ["Chest pain and shortness of breath"]
})

# Convert Symptoms using TF-IDF
patient_symptoms_tfidf = tfidf.transform(patient_data["Symptoms"]).toarray()

# Drop text column and add transformed features
patient_data = patient_data.drop(columns=["Symptoms"])
patient_data = pd.concat([patient_data, pd.DataFrame(patient_symptoms_tfidf, 
                                                      columns=[f"symptom_{i}" for i in range(patient_symptoms_tfidf.shape[1])])], axis=1)

# Scale numerical features
numeric_cols = ["Age", "Consultation_Duration_Minutes", "BMI", "Past_Consultations"]
patient_data[numeric_cols] = scaler.transform(patient_data[numeric_cols])

# Ensure `patient_data` matches `X_train`
missing_cols = set(train_features) - set(patient_data.columns)
extra_cols = set(patient_data.columns) - set(train_features)

# Add missing columns with default values
for col in missing_cols:
    patient_data[col] = 0  

# Remove extra columns that were not in training
patient_data = patient_data[list(train_features)]

# Convert column names to strings
patient_data.columns = patient_data.columns.astype(str)

# Predict required doctor(s)
predicted_specialists = model.predict(patient_data)

# Convert one-hot encoded output to doctor names
all_doctors = preprocessing_objects["doctors"]
doctor_list = [doc for i, doc in enumerate(all_doctors) if predicted_specialists[0][i] == 1]
print(f"Recommended Doctors: {doctor_list}")

# Example doctor availability (This should ideally come from a database)
doctor_availability = {
    "Cardiologist": ["2025-04-02 10:00 AM", "2025-04-02 03:00 PM"],
    "Pulmonologist": ["2025-04-02 11:00 AM", "2025-04-02 04:00 PM"]
}

# Find available slots
for doctor in doctor_list:
    if doctor in doctor_availability:
        print(f"Available slots for {doctor}: {doctor_availability[doctor]}")
    else:
        print(f"No available slots for {doctor} at the moment.")

# Book the first available slot
if doctor_list:
    appointment_details = {
        "Patient_Name": "John Doe",
        "Doctor": doctor_list[0],
        "Appointment_Time": doctor_availability.get(doctor_list[0], ["No Slots Available"])[0]
    }
    print(f"✅ Appointment booked: {appointment_details}")
else:
    print("❌ No available doctors for the given symptoms.")
    
    
#####################################################################################################################

# Example doctor availability (Replace this with a real database if needed)
doctor_availability = {
    "Cardiologist": ["2025-04-02 10:00 AM", "2025-04-02 03:00 PM"],
    "Pulmonologist": ["2025-04-02 11:00 AM", "2025-04-02 04:00 PM"]
}

# Dynamically add missing doctors with random availability
import random
from datetime import datetime, timedelta

def generate_random_slots():
    base_time = datetime(2025, 4, 2, 9, 0)  # Start at 9:00 AM
    slots = [(base_time + timedelta(hours=i)).strftime("%Y-%m-%d %I:%M %p") for i in range(5)]
    return slots

for doctor in doctor_list:
    if doctor not in doctor_availability:
        doctor_availability[doctor] = generate_random_slots()

# Display available slots
for doctor in doctor_list:
    print(f"Available slots for {doctor}: {doctor_availability[doctor]}")

# Book the first available slot
appointment_details = {
    "Patient_Name": "John Doe",
    "Doctor": doctor_list[0],  # First recommended doctor
    "Appointment_Time": doctor_availability[doctor_list[0]][0]  # First available slot
}

print(f"✅ Appointment booked: {appointment_details}")

