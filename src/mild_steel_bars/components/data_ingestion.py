import os
import requests
import gdown
import zipfile
from pathlib import Path
from mild_steel_bars import logger
from mild_steel_bars.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self) -> str:
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            if 'drive.google.com' in dataset_url:
                self.download_from_google_drive(dataset_url, zip_download_dir)
            else:
                self.download_from_raw_url(dataset_url, zip_download_dir)
            
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
        except Exception as e:
            raise e

    def download_from_raw_url(self, url: str, download_dir: Path):
        try:
            logger.info(f'Download data from {url} into file {download_dir}')
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(download_dir, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            logger.info(f'Complete downloading data from {url} into file {download_dir}')

        except Exception as e:
            raise e

    def download_from_google_drive(self, url: str, download_dir: Path):
        try:
            id = url.split('/')[-2]
            gdown.download(f'https://drive.google.com/uc?id={id}', str(download_dir), quiet=False)
            logger.info(f"Complete downloading data from {url} into file {download_dir}")

        except Exception as e:
            raise e

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_file:
            zip_file.extractall(unzip_path)

        logger.info(f"Extracted zip file into {unzip_path}")