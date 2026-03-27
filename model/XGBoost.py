import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier # Thư viện: pip3 install xgboost
from sklearn.metrics import accuracy_score
import joblib
import os
import pandas as pd
def train_xgboost(X_train, y_train, X_test=None, y_test=None,data = "iotid20"):
    os.makedirs("downstream_model", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    # declare parameters
    params = {
                'objective':'???',
                'max_depth': "???",
                'alpha': "???",
                'learning_rate': "???",
                'n_estimators':"???" 
            }
    model = XGBClassifier(**params)

    model.fit( 
                X_train_np, y_train_np,
                eval_set=[(X_train_np, y_train_np)],
                verbose=True  # Hiển thị từng round (boosting iteration))
             )
    joblib.dump(model, "downstream_model/xgboost_" + data + ".pkl")

    if X_test is not None and y_test is not None:
        acc = model.score(X_test.cpu().numpy(), y_test.cpu().numpy())
        print(f"XGBoost accuracy: {acc*100}%")
    return model
    
