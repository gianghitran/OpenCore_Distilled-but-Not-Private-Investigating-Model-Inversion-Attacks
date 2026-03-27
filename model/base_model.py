from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def load(self, path, device):
        pass

    @abstractmethod
    def predict(self, X):
        pass