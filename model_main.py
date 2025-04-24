

from src.preprocessor.data_preprocessor import DataPreprocessor
from src.generator.patient_data_generator import PatientDataGenerator
from src.eda import eda_analysis
from src.trainer.model_trainer import  ModelTrainer
from src.utils.logging_config import logger
from src.utils.config import Config



# Execute all EDA functions
if __name__ == "__main__":

    try:

        logger.info("Starting Project ...")

        # Generate data
        logger.info(f"Generating \"{Config.NUMBER_OF_PATIENT_RECORDS}\" Synthetic Patient Data ...")
        patient_data_generator = PatientDataGenerator()
        dataframe = patient_data_generator.generate_synthetic_data(n_records=Config.NUMBER_OF_PATIENT_RECORDS,output_file=Config.SYNTHETIC_DATA )

        data_preprocessor = DataPreprocessor()
        X, y, scaler, tfidf, le_dict, selector, all_doctors = data_preprocessor.preprocess_data(dataframe)

        logger.info("Starting Visual EDA Analysis...")
        eda_analysis.run_eda()
        logger.info("Stoped Visual EDA Analysis Successfully.")

        # Create an instance of the trainer
        trainer = ModelTrainer()

        # Train and evaluate the model
        model = trainer.train_and_evaluate(X, y, scaler, tfidf, le_dict, selector)

        logger.info("Successfully Completed Model Training and Testing.\n")
    except Exception as e:
        logger.error(f"Pipeline failed due to: {e}")