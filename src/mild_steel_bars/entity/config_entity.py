from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    required_file: list
    data_ingestion_root_dir: Path

@dataclass(frozen=True)
class ModelTrainingAndEvaluationConfig:
    #   Training
    root_dir: Path
    training_data: Path
    data_ingestion_root_dir: Path
    validation_status_file_path: Path
    params_weight: str
    params_batch_size: int
    params_imgsz: int
    params_epoch: int
    params_project: Path

    #   Evaluation
    mlflow_uri: str
    repo_owner: str
    repo_name: str
    params_exp_name: str


@dataclass(frozen=True)
class HyperparameterTuningAndEvaluationConfig:
    #   Hyperparameters
    root_dir: Path
    tuning_data: Path
    data_ingestion_root_dir: Path
    validation_status_file_path: Path

    params_weights: str
    params_batch_size: int
    params_imgsz: int
    params_epoch: int  
    params_project: str 
    params_iterations: int
    params_name: str
    
  

    #   Evaluation
    mlflow_uri: str
    repo_owner: str
    repo_name: str
    params_exp_name: str