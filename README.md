# ai-based-patient-appointment
This project aims to develop an AI-powered system for scheduling patient appointments based on medical history, symptoms, and doctor availability. It includes synthetic dataset generation, exploratory data analysis (EDA), and a machine learning model to predict consultation chains, optimizing healthcare workflow for efficient patient care. 

<pre>
2025-03-27 19:38:21,768 - INFO ->>> --------------------------------------------------------------------------------
2025-03-27 19:38:21,768 - INFO ->>> Starting Project ...
2025-03-27 19:38:21,768 - INFO ->>> Generating "10000" Synthetic Patient Data ...
2025-03-27 19:38:26,037 - INFO ->>> Dataset with 10000 records generated and saved to synthetic_patient_data.csv!
2025-03-27 19:38:51,908 - INFO ->>> Best Parameters: {'estimator__max_depth': 7, 'estimator__min_samples_leaf': 7, 'estimator__min_samples_split': 10, 'estimator__n_estimators': 100}
2025-03-27 19:39:02,433 - INFO ->>> Cross-Validation F1 Score: 0.95 (�0.01)
2025-03-27 19:39:02,643 - INFO ->>> Training F1 Score: 0.95
2025-03-27 19:39:02,741 - INFO ->>> Test Accuracy: 0.88
2025-03-27 19:39:02,742 - INFO ->>> Test F1 Score: 0.94
2025-03-27 19:39:02,742 - INFO ->>> Test Classification Report:
2025-03-27 19:39:02,743 - INFO ->>>                     precision    recall  f1-score   support

      Cardiologist       0.92      0.88      0.90       544
     Dermatologist       1.00      1.00      1.00       161
   Endocrinologist       0.94      1.00      0.97       251
Gastroenterologist       0.98      0.98      0.98       207
       Neurologist       0.78      1.00      0.88       204
      Psychiatrist       0.88      1.00      0.94       145
     Pulmonologist       0.91      1.00      0.96       393
    Rheumatologist       1.00      1.00      1.00       173

         micro avg       0.92      0.97      0.94      2078
         macro avg       0.93      0.98      0.95      2078
      weighted avg       0.92      0.97      0.94      2078
       samples avg       0.95      0.98      0.96      2078

2025-03-27 19:39:02,922 - INFO ->>> Model and preprocessing objects saved!
2025-03-27 19:39:02,922 - INFO ->>> Successfully Completed Model Training and Testing...

2025-03-27 19:39:11,958 - INFO ->>> Starting patient appointment demo...
2025-03-27 19:39:12,881 - INFO ->>> Predicted Doctors: ['Cardiologist']
2025-03-27 19:39:12,918 - INFO ->>> Appointment Details:
2025-03-27 19:39:12,918 - INFO ->>> Patient_Name: Mary Johnson
2025-03-27 19:39:12,918 - INFO ->>> Required_Doctors: ['Cardiologist']
2025-03-27 19:39:12,918 - INFO ->>> Appointment_Date: 2025-03-27
2025-03-27 19:39:12,918 - INFO ->>> Consultation_Type: In-person
2025-03-27 19:39:12,918 - INFO ->>> Hospital_Clinic_Name: City Hospital
2025-03-27 19:39:12,951 - INFO ->>> Predicted Doctors: ['Pulmonologist']
2025-03-27 19:39:12,985 - INFO ->>> Appointment Details:
2025-03-27 19:39:12,985 - INFO ->>> Patient_Name: Robert Patel
2025-03-27 19:39:12,985 - INFO ->>> Required_Doctors: ['Pulmonologist']
2025-03-27 19:39:12,985 - INFO ->>> Appointment_Date: 2025-03-27
2025-03-27 19:39:12,985 - INFO ->>> Consultation_Type: Online
2025-03-27 19:39:12,985 - INFO ->>> Hospital_Clinic_Name: Health Clinic
2025-03-27 19:39:13,034 - INFO ->>> Predicted Doctors: ['Neurologist']
2025-03-27 19:39:13,068 - INFO ->>> Appointment Details:
2025-03-27 19:39:13,068 - INFO ->>> Patient_Name: Sarah Kim
2025-03-27 19:39:13,068 - INFO ->>> Required_Doctors: ['Neurologist']
2025-03-27 19:39:13,068 - INFO ->>> Appointment_Date: 2025-03-27
2025-03-27 19:39:13,068 - INFO ->>> Consultation_Type: Online
2025-03-27 19:39:13,068 - INFO ->>> Hospital_Clinic_Name: Wellness Center
2025-03-27 19:39:13,068 - INFO ->>> Successfully completed patient appointment demo ...

</pre>
