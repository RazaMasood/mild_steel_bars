from ultralytics import YOLO
from ultralytics import settings
import yaml
import mlflow
from urllib.parse import urlparse
import dagshub

from mild_steel_bars import logger
from mild_steel_bars.entity.config_entity import HyperparameterTuningAndEvaluationConfig


class HyperparameterTuningAndEvaluation:
    def __init__(self, config: HyperparameterTuningAndEvaluationConfig):
        self.config = config

    def check_validation_status(self) -> bool:
    # Check validation status before proceeding
        logger.info(f"Checking data validation status")
        try:
            with open(self.config.validation_status_file_path, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    if "Validation status" in line:
                        return "True" in line
            return False
        except Exception as e:
            logger.error(f"Error reading validation status: {e}")
            raise e
        
    def tuningAndEvaluation(self):

        validation_status = self.check_validation_status()

        if validation_status:
            logger.info(f"Data validation completed. Proceeding with hyperparameter tuning and evaluation.")

            logger.info(f"Updating data.yaml file")

            with open(self.config.tuning_data, 'r') as file:
                data_yaml = yaml.safe_load(file)
            # Update paths in train method
            data_yaml['train'] = str(self.config.data_ingestion_root_dir / 'train' / 'images')  
            data_yaml['val'] = str(self.config.data_ingestion_root_dir / 'valid' / 'images')   
            if 'test' in data_yaml:
                data_yaml['test'] = str(self.config.data_ingestion_root_dir / 'test' / 'images')  

            with open(self.config.tuning_data, 'w') as file:
                yaml.dump(data_yaml, file)
            
            logger.info(f"updated data.yaml file before training")

            logger.info(f"Starting model training")

            model = YOLO(self.config.params_weights)
            settings.reset()
            settings.update({
                'mlflow': True,
                'clearml': False,
                'comet': False,
                'dvc': False,
                'hub': False,
                'neptune': False,
                'tensorboard': False,
                'wandb': False
            })

            dagshub.init(repo_owner=self.config.repo_owner, repo_name=self.config.repo_name, mlflow=True)

            # Set the MLflow tracking URI to point to DagsHub
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment(self.config.params_exp_name)  
            # Start a new MLflow run
            with mlflow.start_run():

                # Start tuning hyperparameters for YOLO11n training on the COCO8 dataset
                result_grid = model.tune(
                    data=self.config.tuning_data,
                    batch=self.config.params_batch_size,
                    epochs=self.config.params_epoch,
                    imgsz=self.config.params_imgsz,
                    use_ray=True,
                    iterations=self.config.params_iterations,
                    project=self.config.params_project,
                    name=self.config.params_name
     
                )
            
            logger.info(f"Hyperparameter tuning completed and logged to DagsHub and MLflow.")

            mlflow.end_run()

            return model

        else:
            logger.info(f"Data validation not completed. Skipping hyperparameter tuning and evaluation.")
            return None