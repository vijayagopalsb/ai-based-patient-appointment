# File: synthetic_data/eda_analysis.py
# -----------------------------------------

###########################################################################
# Exploratory Data Analysis (EDA) on Synthetic Patient Dataset           #
# Analyzes patient distribution, disease correlations, and doctor workload #
###########################################################################

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("synthetic_patient_data.csv")

# Set Seaborn style for better visualization
sns.set_style("whitegrid")

# ------------------------------------
# 1. Patient Demographics Analysis
# ------------------------------------

def plot_age_distribution():
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Age"], bins=20, kde=True, color='blue')
    plt.title("Age Distribution of Patients")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()


def plot_gender_distribution():
    plt.figure(figsize=(6, 6))
    df["Gender"].value_counts().plot.pie(autopct="%1.1f%%", colors=["skyblue", "lightcoral", "lightgreen"])
    plt.title("Gender Distribution")
    plt.ylabel("")
    plt.show()
    
# ------------------------------------
# 2. Disease Frequency Analysis
# ------------------------------------

def plot_disease_distribution():
    plt.figure(figsize=(12, 6))
    sns.countplot(y=df["Existing_Disease"], order=df["Existing_Disease"].value_counts().index, palette="coolwarm")
    plt.title("Most Common Diseases")
    plt.xlabel("Count")
    plt.ylabel("Disease")
    plt.show()
    
# ------------------------------------
# 3. Correlation Heatmap
# ------------------------------------

def plot_correlation_heatmap():
    plt.figure(figsize=(10, 6))
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])  # Select only numerical features
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.show()

    
# ------------------------------------
# 4. Doctor Workload Analysis
# ------------------------------------

def plot_doctor_workload():
    plt.figure(figsize=(12, 6))
    df["Doctors_Required"].explode().value_counts().plot(kind="bar", color="teal")
    plt.title("Doctor Workload Analysis")
    plt.xlabel("Specialization")
    plt.ylabel("Number of Appointments")
    plt.xticks(rotation=45)
    plt.show()
    
# Execute all EDA functions
if __name__ == "__main__":
    plot_age_distribution()
    plot_gender_distribution()
    plot_disease_distribution()
    plot_correlation_heatmap()
    plot_doctor_workload()
    
