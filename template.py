import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

list_of_files = [
    'src/__init__.py',
    'src/components/__init__.py',
    'src/components/data_ingestion.py',
    'src/components/data_transformation.py',
    'src/components/model_trainer.py',
    'src/pipelines/__init__.py',
    'src/pipelines/training_pipeline.py',
    'src/pipelines/prediction_pipeline.py',
    'src/utils.py',
    'src/exception.py',
    'src/logger.py',
    'app.py',
    'dockerfile',
    'requirements.txt',
    'setup.py'

]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creating {filedir} for the {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")

