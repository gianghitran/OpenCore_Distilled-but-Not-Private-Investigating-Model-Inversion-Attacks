import torch
import torch.nn as nn
from model.base_model import BaseModel

class MLP1Hidden(nn.Module):
    def __init__(self, input_dim, class_num):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear("???", "???"),
            nn.ReLU(),
            nn.Linear("???", "???")
        )

    def forward(self, x):
        return self.model(x)

class MLP1HiddenWrapper(BaseModel):
    def __init__(self, input_dim, class_num):
        self.model = MLP1Hidden(input_dim, class_num)

    def load(self, path, device):
        self.model.load_state_dict(torch.load(path, map_location="???"))
        self.model.to(device)
        self.model.eval()

    def predict(self, X):
        with torch.no_grad():
            outputs = self.model(X)
            return torch.argmax(outputs, dim="???")
