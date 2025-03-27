# File: model_training/prepare_training_data.py
# ----------------------------------------------------
# Prepares the dataset for training by encoding categorical variables
# and handling missing values where necessary.
# ----------------------------------------------------



# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Load the dataset
df = pd.read_csv("synthetic_patient_data.csv")

# Drop unnecessary columns that do not contribute to model training
df = df.drop(columns=["Patient_ID", "Name", "Address", "Contact_Number", "Email", "Preferred_Doctors", "Hospital_Clinic_Name"])

# Remove duplicate records to prevent overfitting
df = df.drop_duplicates()

# Encode categorical variables using Label Encoding
label_encoders = {}
categorical_columns = ["Gender", "Blood_Group", "Insurance_Provider", "Existing_Disease", "Allergies", 
                       "Chronic_Conditions", "Previous_Surgeries", "Medication_History", "Family_History", 
                       "Doctor_Availability", "Consultation_Type", "Disease_Severity_Level", 
                       "Lab_Test_Results", "Smoking_Alcohol_Use", "Physical_Activity_Level", "Dietary_Preference", 
                       "Mental_Health_Conditions"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for future decoding
    
# Convert multi-label "Doctors_Required" into One-Hot Encoding
one_hot = OneHotEncoder()
doc_encoded = one_hot.fit_transform(df[["Doctors_Required"]]).toarray()
doc_columns = one_hot.get_feature_names_out(["Doctors_Required"])
df_docs = pd.DataFrame(doc_encoded, columns=doc_columns)

# Merge the one-hot encoded columns back into the main dataset
df = df.drop(columns=["Doctors_Required"])
df = pd.concat([df, df_docs], axis=1)

# Handle textual data (Symptoms)
tfidf = TfidfVectorizer(max_features=50)  # Reduce TF-IDF features to limit overfitting
X_symptoms = tfidf.fit_transform(df["Symptoms"].fillna(""))  # Ensure no NaN values

# Convert to DataFrame and merge with the main dataset
df_symptoms = pd.DataFrame(X_symptoms.toarray(), columns=[f"symptom_{i}" for i in range(X_symptoms.shape[1])])
df = df.drop(columns=["Symptoms"])
df = pd.concat([df, df_symptoms], axis=1)

# Convert date columns into numerical format
date_columns = ["Appointment_Date", "Diagnosis_Date"]
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")  # Convert to datetime
        df[col] = (df[col] - datetime(2000, 1, 1)).dt.days  # Convert to number of days since year 2000

# Use 'Doctors_Required' as target variable instead of 'Appointment_Date'
y = df[doc_columns].astype(int)  # Ensure labels are integer type
X = df.drop(columns=date_columns)  # Drop date columns from features

# Ensure 'Doctors_Required' is not in features (to prevent data leakage)
if "Doctors_Required" in X.columns:
    X = X.drop(columns=["Doctors_Required"])

# Shuffle dataset to ensure varied train-test split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Save processed data for model training
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Training data prepared and saved successfully!")