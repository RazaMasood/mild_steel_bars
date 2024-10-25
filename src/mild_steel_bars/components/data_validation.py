import os
from pathlib import Path
from mild_steel_bars import logger
from mild_steel_bars.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            logger.info(f"Validating all files exist in {self.config.data_ingestion_root_dir}")

            os.makedirs(self.config.root_dir, exist_ok=True)

            validation_status = True
            missing_files = []

            for required_file in self.config.required_file:
                required_file = Path(self.config.data_ingestion_root_dir) / required_file
                if not required_file.exists():
                    validation_status = False
                    missing_files.append(Path(required_file))

            with open(self.config.status_file, "w") as f:
                f.write(f'Validation status: {validation_status}\n')
                if not validation_status:
                    f.write(f"Missing folder: {','.join(missing_files)}\n")
                else:
                    f.write("All required files are present\n")

            logger.info(f'Validation status: {validation_status}')

        except Exception as e:
            print(f"An error occurred during data validation: {e}")
            raise e
