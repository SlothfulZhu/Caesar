# coding=utf-8
from sklearn.externals import joblib


class ModuleClient:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        model = joblib.load(self.model_path)
        return model

    def invoke(self, record):
        pred = self.model.predict(record)
        return pred
