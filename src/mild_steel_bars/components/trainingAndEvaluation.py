from ultralytics import YOLO
from ultralytics import settings
import yaml
import mlflow
from urllib.parse import urlparse
import dagshub

from mild_steel_bars import logger
from mild_steel_bars.entity.config_entity import ModelTrainingAndEvaluationConfig


class TrainingAndEvaluation:
    def __init__(self, config: ModelTrainingAndEvaluationConfig):
        self.config =config
    
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

    def trainAndEvaluate(self):
        # Check the validation status
        validation_status = self.check_validation_status()

        # Proceed only if validation status is True
        if validation_status:
            logger.info(f"Validation passed. Proceeding with training.")

            logger.info(f"Updating data.yaml file")

            with open(self.config.training_data, 'r') as file:
                data_yaml = yaml.safe_load(file)

            # Update paths in train method
            data_yaml['train'] = str(self.config.data_ingestion_root_dir / 'train' / 'images')  
            data_yaml['val'] = str(self.config.data_ingestion_root_dir / 'valid' / 'images')   
            if 'test' in data_yaml:
                data_yaml['test'] = str(self.config.data_ingestion_root_dir / 'test' / 'images')  

            with open(self.config.training_data, 'w') as file:
                yaml.dump(data_yaml, file)
            
            logger.info(f"updated data.yaml file before training")

            logger.info(f"Starting model training")

            model = YOLO(self.config.params_weight)
            settings.reset()
            settings.update({
                'mlflow': True,
                'clearml': False,
                'comet': False,
                'dvc': False,
                'hub': False,
                'neptune': False,
                'raytune': False,
                'tensorboard': False,
                'wandb': False
            })

            
            dagshub.init(repo_owner=self.config.repo_owner, repo_name=self.config.repo_name, mlflow=True)
            
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            mlflow.set_experiment(self.config.params_exp_name)

            with mlflow.start_run():
                model.train(data=self.config.training_data, epochs=self.config.params_epoch, batch=self.config.params_batch_size, imgsz=self.config.params_imgsz, project=self.config.params_project)

            logger.info(f"Model training completed")
            logger.info(f"Visit you repo for Evaluation")

            mlflow.end_run()

            return model 

        else:
            logger.error(f"Validation failed. Training cannot proceed.")
            return None       