from fastapi import FastAPI, Depends
from sklearn.pipeline import Pipeline
from loguru import logger
import os
import contextlib
from models import PredictionInput, PredictionOutput
import joblib

#basic logger configuration
logger.add("file_{time}.log", rotation="1 week")

class NewsgroupsModel:
    model: Pipeline | None = None
    targets: list[str] | None = None

    def load_model(self) -> None:
        "Load the model"
        logger.info("Loading the model")
        model_file = os.path.join(os.path.dirname(__file__), "newsgroups_models.joblib")

        try:
            loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file)
            self.model, self.targets = loaded_model
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

