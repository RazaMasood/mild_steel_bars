import os
import sys
import argparse

# Add the 'src' folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mild_steel_bars import logger
from mild_steel_bars.pipelines.data_ingestion import DataIngestionTrainingPipeline
from mild_steel_bars.pipelines.data_validation import DataValidationTrainingPipeline
from mild_steel_bars.pipelines.trainingAndEvaluation import ModelTrainingAndEvaluationPipeline
from mild_steel_bars.pipelines.tuningAndEvaluation import ModelTuningAndEvaluationPipeline



def run_data_ingestion():
   STAGE_NAME = "Data Ingestion Stage"
   try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
      data_ingestion = DataIngestionTrainingPipeline()
      data_ingestion.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   except Exception as e:
         logger.exception(e)
         raise e
   

def run_data_validation():
   STAGE_NAME = "Data Validation Stage"
   try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
      data_validation = DataValidationTrainingPipeline()
      data_validation.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   except Exception as e:
      logger.exception(e)

def run_model_training():
   STAGE_NAME = "Model Training and Evaluation Stage"
   try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
      train_and_eval = ModelTrainingAndEvaluationPipeline()
      train_and_eval.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   except Exception as e:
      logger.exception(e)


def run_hyperparameter_tuning():
   STAGE_NAME = "Model Tuning and Evaluation Stage"
   try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
      tune_and_eval = ModelTuningAndEvaluationPipeline()
      tune_and_eval.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   except Exception as e:
      logger.exception(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Operations: Choose between training or hyperparameter tuning.")

    # Argument to specify either training or hyperparameter tuning
    parser.add_argument(
        "--stage", 
        type=str, 
        required=True, 
        choices=["train", "tune"],
        help="Choose 'train' to run model training or 'tune' for hyperparameter tuning."
    )

    args = parser.parse_args()

    # Run data ingestion and validation (these will always run)
    run_data_ingestion()
    run_data_validation()

    # Run model training or hyperparameter tuning based on user's input
    if args.stage == "train":
        run_model_training()
    elif args.stage == "tune":
        run_hyperparameter_tuning()