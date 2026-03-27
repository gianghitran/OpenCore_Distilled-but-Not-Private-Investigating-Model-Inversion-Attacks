import joblib
from model.base_model import BaseModel

class SklearnModelWrapper(BaseModel):
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None

    def load(self, path, _):
        self.model = joblib.load(path)

    def predict(self, X):
        return self.model.predict(X)