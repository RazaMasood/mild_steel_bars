from mild_steel_bars.constants import *
import os
from mild_steel_bars.utils.main_utils import read_yaml_file, create_directories
from mild_steel_bars.entity.config_entity import (DataIngestionConfig, 
                                                  DataValidationConfig, 
                                                  ModelTrainingAndEvaluationConfig,
                                                  HyperparameterTuningAndEvaluationConfig)


class ConfigurationManager:
    def __init__(self,
                config_filepath=CONFIG_FILE_PATH,
                params_filepath=PARAMS_FILE_PATH):
        
        self.config = read_yaml_file(config_filepath)
        self.params = read_yaml_file(params_filepath)

        create_directories([self.config['artifacts_root']]) 

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']  
        create_directories([config['root_dir']]) 

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            source_URL=config['source_URL'], 
            local_data_file=Path(config['local_data_file']),  
            unzip_dir=Path(config['unzip_dir']),  
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config["data_validation"]
        create_directories([config["root_dir"]])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config['root_dir']),
            status_file=Path(config['status_file']),
            required_file=config['required_file'],
            data_ingestion_root_dir=Path(config['data_ingestion_root_dir']),
        )
    
        return data_validation_config
    
        
    def get_model_trainer_and_evaluation_config(self) -> ModelTrainingAndEvaluationConfig:
        train = self.config['model_training']
        eval = self.config['model_evaluation']        
        params = self.params['TRAINING_PARAMETERS']
        create_directories([train['root_dir']])

        training_config = ModelTrainingAndEvaluationConfig(
            root_dir=Path(train['root_dir']),
            training_data=Path(train['training_data']),
            data_ingestion_root_dir=Path(train['data_ingestion_root_dir']).resolve(),
            validation_status_file_path=Path(train['validation_status_file_path']),
            params_weight=params['PRETRAINED_MODEL'],
            params_batch_size=params['BATCH_SIZE'],
            params_imgsz=params['IMAGE_SIZE'],
            params_epoch=params['EPOCH'],
            params_project=params['PROJRCT'],

            repo_owner=eval['repo_owner'],
            repo_name=eval['repo_name'],
            mlflow_uri=eval['mlflow_uri'],
            params_exp_name=params['EXPERIMENT_NAME']

            
        )

        return training_config
    

    def get_hyperparameter_tuning_and_evaluation_config(self) -> HyperparameterTuningAndEvaluationConfig:
        hyp = self.config['hyperparameter_tuning']
        eval = self.config['model_evaluation']        
        params = self.params['HYPERPARAMETER_TUNING_PARAMETERS']
        create_directories([hyp['root_dir']])

        tuning_config = HyperparameterTuningAndEvaluationConfig(
            root_dir=Path(hyp['root_dir']),
            tuning_data=Path(hyp['tuning_data']).resolve(),
            data_ingestion_root_dir=Path(hyp['data_ingestion_root_dir']).resolve(),
            validation_status_file_path=Path(hyp['validation_status_file_path']),
            params_weights=params['PRETRAINED_MODEL'],
            params_batch_size=params['BATCH_SIZE'],
            params_imgsz=params['IMAGE_SIZE'],
            params_epoch=params['EPOCH'],
            params_project=params['PROJRCT'],
            params_iterations=params['ITERATIONS'],
            params_name=params['NAME'],

            repo_owner=eval['repo_owner'],
            repo_name=eval['repo_name'],
            mlflow_uri=eval['mlflow_uri'],
            params_exp_name=params['EXPERIMENT_NAME']

            
        )

        return tuning_config