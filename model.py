###########################################################################
# Patient Doctor Prediction Pipeline                                      #
# Generates synthetic patient data, preprocesses it, trains a model,      #
# and evaluates performance to predict required doctors.                 #
# Optimized to reduce overfitting with larger data, feature selection,   #
# and hyperparameter tuning.                                             #
###########################################################################

# Import Libraries
import pandas as pd
import numpy as np
import random
from faker import Faker
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Faker for synthetic data
faker = Faker()

# --- Disease and Categorical Definitions ---
diseases = {
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
blood_groups = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
insurance_providers = ["None", "MediCare", "HealthFirst", "Aetna", "Blue Cross"]
consultation_types = ["Online", "In-person"]
severity_levels = ["Mild", "Moderate", "Severe"]
activity_levels = ["Sedentary", "Active", "Highly Active"]
dietary_preferences = ["Vegetarian", "Non-Vegetarian", "Vegan"]

# --- Data Generation ---
def generate_synthetic_data(n_records=10000):
    data = []
    for _ in range(n_records):
        age = random.randint(5, 90)
        # Introduce age-disease correlation
        if age > 60:
            disease = random.choices(list(diseases.keys()), weights=[0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])[0]
        else:
            disease = random.choice(list(diseases.keys()))
        # Multi-label for some diseases (e.g., severe Diabetes)
        specializations = diseases[disease]
        if disease == "Diabetes" and random.random() < 0.3:
            specializations = ["Endocrinologist", "Cardiologist"]

        patient_id = faker.uuid4()
        name = faker.name()
        gender = random.choice(["Male", "Female", "Other"])
        blood_group = random.choice(blood_groups)
        address = faker.address()
        contact_number = faker.phone_number()
        email = faker.email()
        insurance_provider = random.choice(insurance_providers)
        allergies = random.choice(["None", "Peanuts", "Pollen", "Dust", "Penicillin"])
        chronic_conditions = random.choice(["None", "Hypertension", "Diabetes", "Asthma"])
        previous_surgeries = random.choice(["None", "Appendectomy", "Knee Replacement", "Gallbladder Removal"])
        medication_history = random.choice(["None", "Metformin", "Aspirin", "Ibuprofen"])
        family_history = random.choice(["None", "Heart Disease", "Diabetes", "Cancer"])
        past_consultations = random.randint(0, 20)
        preferred_doctor = faker.name()
        appointment_date = faker.date_this_year()
        consultation_duration_minutes = random.choice([15, 30, 45, 60])
        doctor_availability = random.choice(["Available", "Busy", "Limited Slots"])
        consultation_type = random.choice(consultation_types)
        hospital_clinic_name = faker.company()
        symptoms = faker.sentence(nb_words=6)
        disease_severity_level = random.choice(severity_levels)
        diagnosis_date = faker.date_this_decade()
        lab_test_results = random.choice(["Normal", "Abnormal", "Borderline"])
        smoking_alcohol_use = random.choice(["Yes", "No"])
        bmi = round(random.uniform(18.5, 35.0), 1)
        physical_activity_level = random.choice(activity_levels)
        dietary_preference = random.choice(dietary_preferences)
        mental_health_conditions = random.choice(["None", "Anxiety", "Depression", "Bipolar Disorder"])

        data.append([
            patient_id, name, age, gender, blood_group, address, contact_number, email,
            insurance_provider, disease, allergies, chronic_conditions, previous_surgeries,
            medication_history, family_history, past_consultations, specializations, preferred_doctor,
            appointment_date, consultation_duration_minutes, doctor_availability, consultation_type,
            hospital_clinic_name, symptoms, disease_severity_level, diagnosis_date,
            lab_test_results, smoking_alcohol_use, bmi, physical_activity_level, dietary_preference,
            mental_health_conditions
        ])

    columns = [
        "Patient_ID", "Name", "Age", "Gender", "Blood_Group", "Address", "Contact_Number", "Email",
        "Insurance_Provider", "Existing_Disease", "Allergies", "Chronic_Conditions", "Previous_Surgeries",
        "Medication_History", "Family_History", "Past_Consultations", "Doctors_Required", "Preferred_Doctors",
        "Appointment_Date", "Consultation_Duration_Minutes", "Doctor_Availability", "Consultation_Type",
        "Hospital_Clinic_Name", "Symptoms", "Disease_Severity_Level", "Diagnosis_Date",
        "Lab_Test_Results", "Smoking_Alcohol_Use", "BMI", "Physical_Activity_Level", "Dietary_Preference",
        "Mental_Health_Conditions"
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("synthetic_patient_data.csv", index=False, encoding="utf-8")
    print("Dataset generated and saved!")
    return df

# --- Preprocessing ---
def preprocess_data(df):
    # Separate target column (Doctors_Required) since it contains lists
    y_temp = df["Doctors_Required"]
    df_features = df.drop(columns=["Doctors_Required"])

    # Drop unnecessary columns
    cols_to_drop = ["Patient_ID", "Name", "Address", "Contact_Number", "Email", "Preferred_Doctors", "Hospital_Clinic_Name"]
    df_features = df_features.drop(columns=cols_to_drop)

    # Drop duplicates on features only (excluding target)
    df_features = df_features.drop_duplicates()

    # Reattach target column, aligning indices
    df = df_features.join(y_temp.loc[df_features.index])

    # Convert dates to days since 2000
    df["Appointment_Date"] = pd.to_datetime(df["Appointment_Date"]).map(lambda x: (x - pd.Timestamp("2000-01-01")).days)
    df["Diagnosis_Date"] = pd.to_datetime(df["Diagnosis_Date"]).map(lambda x: (x - pd.Timestamp("2000-01-01")).days)

    # Encode categorical variables
    categorical_cols = [
        "Gender", "Blood_Group", "Insurance_Provider", "Existing_Disease", "Allergies",
        "Chronic_Conditions", "Previous_Surgeries", "Medication_History", "Family_History",
        "Doctor_Availability", "Consultation_Type", "Disease_Severity_Level", "Lab_Test_Results",
        "Smoking_Alcohol_Use", "Physical_Activity_Level", "Dietary_Preference", "Mental_Health_Conditions"
    ]
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Scale numeric columns
    numeric_cols = ["Age", "Consultation_Duration_Minutes", "BMI", "Past_Consultations"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # TF-IDF for symptoms
    tfidf = TfidfVectorizer(max_features=20)
    symptoms_tfidf = tfidf.fit_transform(df["Symptoms"].fillna("")).toarray()
    tfidf_cols = [f"symptom_{i}" for i in range(20)]
    df_tfidf = pd.DataFrame(symptoms_tfidf, columns=tfidf_cols)
    df = pd.concat([df.drop(columns=["Symptoms"]), df_tfidf], axis=1)

    # Prepare target (multi-label)
    all_doctors = sorted(set([doc for sublist in df["Doctors_Required"] for doc in sublist]))
    y = pd.DataFrame(0, index=df.index, columns=all_doctors)
    for idx, doctors in enumerate(df["Doctors_Required"]):
        for doc in doctors:
            y.loc[idx, doc] = 1

    # Feature selection with f_classif
    X = df.drop(columns=["Doctors_Required"])
    selector = SelectKBest(f_classif, k=30)
    X_selected = selector.fit_transform(X, y.idxmax(axis=1))  # Proxy for multi-label
    selected_features = X.columns[selector.get_support()].tolist()

    X = pd.DataFrame(X_selected, columns=selected_features)
    return X, y, scaler, tfidf, le_dict, selector, all_doctors

# --- Training and Evaluation ---
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Define and tune model
    rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=42)
    model = MultiOutputClassifier(rf)
    param_grid = {
        'estimator__n_estimators': [50, 100],
        'estimator__max_depth': [3, 5, 7],
        'estimator__min_samples_split': [10, 15],
        'estimator__min_samples_leaf': [5, 7]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_micro', n_jobs=-1)  # Fixed typo: 'f_micro' -> 'f1_micro'
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_micro')
    print(f"Cross-Validation F1 Score: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

    # Training performance
    y_train_pred = model.predict(X_train)
    train_f1 = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)['micro avg']['f1-score']
    print(f"Training F1 Score: {train_f1:.2f}")

    # Test performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_f1 = classification_report(y_test, y_pred, output_dict=True, zero_division=0)['micro avg']['f1-score']
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred, target_names=y.columns, zero_division=0))

    # Save model and preprocessing objects
    joblib.dump(model, "trained_appointment_model.pkl")
    joblib.dump({"scaler": scaler, "tfidf": tfidf, "le_dict": le_dict, "selector": selector, "doctors": y.columns}, "preprocessing_objects.pkl")
    # Save the scaler
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as 'scaler.pkl'")
    # Save the TF-IDF vectorizer
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")
    print("Model and preprocessing objects saved!")

    return model

# --- EDA (Optional Visualization) ---
def plot_doctor_distribution(y):
    doctor_counts = y.sum()
    plt.figure(figsize=(12, 6))
    #sns.barplot(x=doctor_counts.index, y=doctor_counts.values, palette="viridis")
    sns.barplot(x=doctor_counts.index, y=doctor_counts.values, hue=doctor_counts.index, palette="viridis", legend=False)

    plt.title("Doctor Workload Distribution")
    plt.xlabel("Specialization")
    plt.ylabel("Number of Appointments")
    plt.xticks(rotation=45)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Generate data
    df = generate_synthetic_data(n_records=10000)

    # Preprocess
    X, y, scaler, tfidf, le_dict, selector, all_doctors = preprocess_data(df)

    # Train and evaluate
    model = train_and_evaluate(X, y)

    # Visualize doctor distribution
    plot_doctor_distribution(y)