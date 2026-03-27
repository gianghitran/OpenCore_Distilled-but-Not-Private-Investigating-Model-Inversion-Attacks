import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
def train_random_forest(X_train, y_train, X_test=None, y_test=None,data = "iotid20"):
    os.makedirs("downstream_model", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    model = RandomForestClassifier(n_estimators="???", random_state="???")
    model.fit(X_train_np, y_train_np)
    joblib.dump(model, "downstream_model/random_forest_" + data + ".pkl")

    if X_test is not None and y_test is not None:
        acc = model.score(X_test.cpu().numpy(), y_test.cpu().numpy())
        print(f"Random forest accuracy: {acc*100}%")
    return model