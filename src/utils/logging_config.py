# File: Logging_cofig.py
# Step 1: Step 1: Logging Configuration
# Define a centralized logger

import logging
from src.utils.config import Config

# Convert LOG_LEVEL string to logging level integer (e.g., "INFO" -> logging.INFO)
numeric_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)

# Configure Logging
logging.basicConfig(level=numeric_level, 
                    format="%(asctime)s - %(levelname)s -->> %(message)s",
                    handlers=[
                        logging.FileHandler(Config.LOG_FILE), # Save logs to a file
                        logging.StreamHandler() # Print logs to a console
                    ])

logger = logging.getLogger(__name__)