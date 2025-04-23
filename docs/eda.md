[Back to Main README](../README.md)

--- 

## Exploratory Data Analysis (EDA)

This project performs Exploratory Data Analysis (EDA) on a synthetic patient dataset to uncover insights about patient demographics, disease prevalence, doctor workload, and feature correlations. The goal is to analyze trends that can support healthcare management decisions, such as resource allocation, patient profiling, and workload balancing among doctors.

### Dataset
The synthetic_patient_data.csv file includes:

- Age: Patient's age.

- Gender: Male, Female, or Other.

- Existing_Disease: Patient's pre-existing condition(s).

- Doctors_Required: Specialist(s) required for consultation.

- Past_Consultations: Number of past consultations.

- Duration_Minutes: Duration of consultation in minutes.

- BMI: Body Mass Index.

### EDA Components

1. Age Distribution

![Age Distribution](/output_images/eda/age_distribution.png)

- Analyzes the distribution of patient ages.
- Includes histogram with a KDE (Kernel Density Estimation) curve.

2. Gender Distribution

![Gender Distribution](/output_images/eda/gender_distribution.png)

- Pie chart showing the proportion of Male, Female, and Other genders.

3. Disease Frequency Distribution

![Disease Distribution](/output_images/eda/disease_distribution.png)

- Horizontal bar chart showing the most common diseases in the dataset.

4. Doctor Workload Analysis

![Doctor Workload](/output_images/eda/dr_workload.png)

- Bar chart displaying the number of appointments per specialist type.

- Helps identify overburdened specializations like Cardiologists and Pulmonologists.

5. Correlation Heatmap

![Correlation Heatmap](/output_images/eda/heatmap.png)

- Heatmap displaying correlations between numerical features such as:

1. Age

2. BMI

3. Past Consultations

4. Consultation Duration


---

[Back to Main README](../README.md)
