# File: model_main.py

###########################################################################
# Patient Doctor Prediction Pipeline                                      #
# Generates synthetic patient data, preprocesses it, trains a model,      #
# and evaluates performance to predict required doctors.                  #
# Optimized to reduce overfitting with larger data, feature selection,    #
# and hyperparameter tuning.                                              #
###########################################################################

# Import Libraries
import pandas as pd
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from faker import Faker
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report

# Import App Libraries
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessor.data_preprocessor import DataPreprocessor  # Absolute import   
from data_generator.patient_data_generator import PatientDataGenerator
from data_preprocessor.data_preprocessor import DataPreprocessor
from model_trainer.model_trainer import ModelTrainer
from logging_config import logger

# --- Main Execution ---
if __name__ == "__main__":
    
    NUMBER_OF_PATIENT_RECORDS = 10000
    
    logger.info("-"*80)
    logger.info("Starting Project ...")

    # Generate data
    logger.info(f"Generating \"{NUMBER_OF_PATIENT_RECORDS}\" Synthetic Patient Data ...")
    patient_data_generator = PatientDataGenerator()
    dataframe = patient_data_generator.generate_synthetic_data(NUMBER_OF_PATIENT_RECORDS)
    
    data_preprocessor = DataPreprocessor()
    X, y, scaler, tfidf, le_dict, selector, all_doctors = data_preprocessor.preprocess_data(dataframe)

    # Create an instance of the trainer
    trainer = ModelTrainer()
    
    # Train and evaluate the model
    model = trainer.train_and_evaluate(X, y, scaler, tfidf, le_dict, selector)
    
    logger.info("Successfully Completed Model Training and Testing...\n")