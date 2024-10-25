import os.path
from box.exceptions import BoxValueError
import yaml
import base64
from pathlib import Path

from src.mild_steel_bars import logger


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as yaml_file:
            logger.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)
    
    except BoxValueError:
        raise ValueError("yaml file is empty")
    
    except Exception as e:
        raise e

def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        Path(path).mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at {path}")


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, 'rb') as f:
        return base64.b64encode(f.read())