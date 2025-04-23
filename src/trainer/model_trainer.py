# File: model_trainer/model_trainer.py

###########################################################################
# Model Trainer Class                                                     
#                                                                         
# This module defines a class to train and evaluate a multi-label         
# classification model for predicting required doctors based on patient  
# data. It uses a RandomForestClassifier with GridSearchCV for tuning.   
###########################################################################

import pandas as pd
import joblib
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Import your preprocessor class
#from ..preprocessor.data_preprocessor import DataPreprocessor

# Import App Libraries
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessor.data_preprocessor import DataPreprocessor  # Absolute import  
from  utils.logging_config import logger

class ModelTrainer:
    """
    A class to train and evaluate a multi-label classification model for doctor prediction.

    """

    def __init__(self):
        """Initialize the ModelTrainer with no initial state."""
        pass  # No initialization needed; all model setup occurs in the method

    def train_and_evaluate(self, X, y, scaler=None, tfidf=None, le_dict=None, selector=None):
        """
        Train and evaluate a multi-label classification model on patient data.
        """
        # --- Data Splitting ---
        # Split the data into training (80%) and test (20%) sets with a fixed random seed for reproducibility
        logger.info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        logger.info("Data split completed.")

        # --- Model Definition and Tuning ---
        # Define a RandomForestClassifier with balanced class weights to handle imbalanced data
        rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=42)
        # Wrap it in MultiOutputClassifier for multi-label prediction
        model = MultiOutputClassifier(rf)
        # Define hyperparameter grid for tuning
        
        param_grid = {
            'estimator__n_estimators': [50, 100],      # Number of trees in the forest
            'estimator__max_depth': [3, 5, 7],         # Maximum depth of each tree
            'estimator__min_samples_split': [10, 15],  # Minimum samples to split a node
            'estimator__min_samples_leaf': [5, 7]      # Minimum samples per leaf
        }
        
        logger.info("Starting GridSearchCV for hyperparameter tuning...")
        # Perform grid search with 5-fold cross-validation, optimizing for micro F1 score
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_micro', n_jobs=-1, verbose=2)
        logger.info("Starting hyperparameter tuning using GridSearchCV")
        grid_search.fit(X_train, y_train)  # Fit the grid search to find the best model

        logger.info(f"GridSearchCV Completed. Best Parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_  # Use the best model found
        logger.info(f"Best Parameters: {grid_search.best_params_}")  # Display the optimal hyperparameters
        logger.info("Grid search completed.")

        # --- Cross-Validation ---
        # Evaluate the model with 5-fold cross-validation on the training set
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_micro')
        logger.info(f"Cross-Validation F1 Score: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")  # Mean and std of F1 scores
        logger.info("Cross-validation completed.")
        # --- Training Performance ---
        # Predict on the training set to assess overfitting
        logger.info("Evaluating training and test performance...")
        y_train_pred = model.predict(X_train)
        # Calculate micro-averaged F1 score for training data, handling zero division
        train_f1 = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)['micro avg']['f1-score']
        logger.info(f"Training F1 Score: {train_f1:.2f}")

        # --- Test Performance ---
        # Predict on the test set to evaluate generalization
        y_pred = model.predict(X_test)
        # Calculate accuracy (exact match across all labels)
        accuracy = accuracy_score(y_test, y_pred)
        # Calculate micro-averaged F1 score for test data
        test_f1 = classification_report(y_test, y_pred, output_dict=True, zero_division=0)['micro avg']['f1-score']
        logger.info(f"Test Accuracy: {accuracy:.2f}")
        logger.info(f"Test F1 Score: {test_f1:.2f}")
        # Print detailed classification report for each doctor specialization
        logger.info("Test Classification Report:")
        logger.info("\n"+classification_report(y_test, y_pred, target_names=y.columns, zero_division=0))

        # --- Save Model and Preprocessing Objects ---
        # Save the trained model to a file
        joblib.dump(model, "models/trained_appointment_model.pkl")
        # Save preprocessing objects in a single file, assuming they are passed as arguments
        logger.info("Evaluation completed. Saving model...")
        if scaler and tfidf and le_dict and selector:
            joblib.dump({
                "scaler": scaler,
                "tfidf": tfidf,
                "le_dict": le_dict,
                "selector": selector,
                "doctors": y.columns
            }, "models/preprocessing_objects.pkl")
            logger.info("Model and preprocessing objects saved to 'models' directory!")
        else:
            logger.info("Model saved! Preprocessing objects not saved due to missing arguments.")

        return model


# Example usage (can be removed in production)
if __name__ == "__main__":
    
    # Load synthetic data and preprocess it (assuming prior preprocessing)
    df = pd.read_csv("synthetic_patient_data.csv")
    
    data_preprocessor = DataPreprocessor()
    X, y, scaler, tfidf, le_dict, selector, all_doctors = data_preprocessor.preprocess_data(df)

    # Create an instance of the trainer
    trainer = ModelTrainer()
    
    # Train and evaluate the model
    model = trainer.train_and_evaluate(X, y, scaler, tfidf, le_dict, selector)
