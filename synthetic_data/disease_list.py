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

