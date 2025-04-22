# File: Logging_cofig.py
# Step 1: Step 1: Logging Configuration
# Define a centralized logger

import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s ->>> %(message)s",
                    handlers=[
                        logging.FileHandler("logs/ai_patient_appointment.log"), # Save logs to a file
                        logging.StreamHandler() # Print logs to a console
                    ])

logger = logging.getLogger(__name__)