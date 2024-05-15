"""
This code demonstrate how to train a XGBoost Logistic Regression model for credit card fraud detection
The code use datasets from 3 parties
- 2 banks providing the labels (class) for each transactions being fraudulent or not
- A financial data intermediary or payment processor providing the transactions data on which Dimensionality Reduction Techniques for Data Privacy has been applied.

"""


import logging
import time
import requests
import os
import json

import duckdb

from dv_utils import default_settings, Client 

import pandas as pd
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report  # Import accuracy_score
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

input_dir = "/resources/data"
output_dir = "/resources/outputs"

# let the log go to stdout, as it will be captured by the cage operator
logging.basicConfig(
    level=default_settings.log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# define an event processing function
def event_processor(evt: dict):
    """
    Process an incoming event
    Exception raised by this function are handled by the default event listener and reported in the logs.
    """
    
    logger.info(f"Processing event {evt}")

    # dispatch events according to their type
    evt_type =evt.get("type", "")
    if(evt_type == "TRAIN"):
        process_train_event(evt)
    else:
        generic_event_processor(evt)


def generic_event_processor(evt: dict):
    # push an audit log to reccord for an event that is not understood
    logger.info(f"Received an unhandled event {evt}")

def process_train_event(evt: dict):
    """
    Train an XGBoost Classifier model using the logic given in 
     """

    logger.info(f"--------------------------------------------------")
    logger.info(f"|               START TRAINING                   |")
    logger.info(f"|                                                |")
    # load the training data from data providers
    # duckDB is used to load the data and aggregated them in one single datasets
    logger.info(f"| 1. Load data from data providers               |")
    logger.info(f"|    https://github.com/./demographic.parquet |")
    logger.info(f"|    https://github.com/./patients.parquet |")
    dataProvider1URL="https://github.com/datavillage-me/cage-process-clinical-trial-patient-cohort-selection/raw/main/data/demographic.parquet"
    #dataProvider1URL="data/demographic.parquet"
    dataProvider2URL="https://github.com/datavillage-me/cage-process-clinical-trial-patient-cohort-selection/raw/main/data/patients.parquet"
    #dataProvider2URL="data/patients.parquet"
    dataProvider3URL="https://github.com/datavillage-me/cage-process-clinical-trial-result-prediction/raw/main/data/outcome.parquet"
    #dataProvider3URL="data/outcome.parquet"
    start_time = time.time()
    logger.info(f"|    Start time:  {start_time} secs |")
    df = duckdb.sql("SELECT *  from '"+dataProvider3URL+"' as outcome,'"+dataProvider1URL+"' as demographic,'"+dataProvider2URL+"' as patients WHERE demographic.national_id=patients.national_id AND demographic.national_id=outcome.national_id").df()
    
    execution_time=(time.time() - start_time)
    logger.info(f"|    Execution time:  {execution_time} secs |")

    logger.info(f"|                                                |")
    logger.info(f"| 2. Create Classification      model            |")

    # Define features
    demographic_features = ["age", "gender", "location", "income_level", "education_level", "employment_status"]
    medical_features = ["medical_problem", "medical_medication", "medical_vaccine"]
    target = "clinical_trial_outcome"
    
    # Encode categorical variables
    merged_df = pd.get_dummies(df, columns=[ "income_level", "education_level", "employment_status"])
    # Combine numerical and encoded categorical features
    features = merged_df.columns.drop([target,'national_id','national_id_1','national_id_2',"medical_problem", "medical_medication", "medical_vaccine", "location","gender"])


    #print(merged_df.columns)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(merged_df[features], merged_df[target], test_size=0.2, random_state=42)

    logger.info(f"|    Create XGBClassifier model                  |")
    # Initialize and train XGBoost classifier
    model = XGBClassifier()
    logger.info(f"|    Run model                                   |")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    classificationReportJson=classification_report(y_test, y_pred,output_dict=True)

    logger.info(f"|                                                |")
    logger.info(f"| 3. Save outputs of the collaboration           |")
    logger.info(f"|    Save created model                          |")
    logger.info(f"|    Save model classification report            |")
    # save the model to the results location
    model.save_model('/resources/outputs/model.json')
    
    with open('/resources/outputs/classification-report.json', 'w', newline='') as file:
       file.write(json.dumps(classificationReportJson))

    logger.info(f"|                                                |")
    logger.info(f"--------------------------------------------------")
   

if __name__ == "__main__":
    test_event = {
            "type": "TRAIN"
    }
    process_train_event(test_event)