# File: synthetic_data/disease_list.py
# ------------------------------------
# Dictionary mapping diseases to their respective specialist doctors
# This allows easy lookup for required medical consultations based on patient conditions.

diseases = {
    "Hypertension": ["Cardiologist"],  # High blood pressure needs a heart specialist
    "Diabetes": ["Endocrinologist"],  # Diabetes management falls under hormonal specialists
    "Asthma": ["Pulmonologist"],  # Lung and respiratory issues require a lung specialist
    "Migraine": ["Neurologist"],  # Chronic headaches are treated by a brain specialist
    "Arthritis": ["Rheumatologist"],  # Joint pain and autoimmune diseases need a specialist
    "Heart Disease": ["Cardiologist"],  # General heart-related issues handled by a cardiologist
    "Pneumonia": ["Pulmonologist"],  # Lung infection requiring a lung specialist
    "Depression": ["Psychiatrist"],  # Mental health conditions need a psychiatrist
    "Skin Allergy": ["Dermatologist"],  # Skin conditions and allergies treated by a skin specialist
    "Gastroenteritis": ["Gastroenterologist"]  # Stomach and digestive issues treated by a specialist
}

# It is very easy to add more diseases and their corresponding specialists here
# Simply follow the format: "Disease Name": ["Specialist"]

# Define possible values for categorical fields
genders = ["Male", "Female", "Other"]
blood_groups = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
consultation_types = ["Online", "In-person"]
severity_levels = ["Mild", "Moderate", "Severe"]
activity_levels = ["Sedentary", "Active", "Highly Active"]
dietary_preferences = ["Vegetarian", "Non-Vegetarian", "Vegan"]
insurance_providers = ["None", "MediCare", "HealthFirst", "Aetna", "Blue Cross"]

