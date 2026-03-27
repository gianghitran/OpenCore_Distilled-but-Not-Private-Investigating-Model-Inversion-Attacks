# factory.py: tạo các model TeacherModel, StudentModel qua ModelFactory.
# evaluation.py: gọi luồng chính.
# preprocess.py: xử lý dữ liệu đầu vào.
# train.py: huấn luyện model.
# evaluate.py: đánh giá và ghi báo cáo.


import pandas as pd
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from utils.saveReport import save_report
from model.MLP import MLP
from model.CNN1D import CNN1D
from model.ResNet import ResNetTabular
from model.core_model_factory import CoreModelFactory
import wandb
import os

def evaluation(X_test, y_test, class_num, input_dir, save_name, report, model_type="MLP", studentOrTeacher="student", num_hidden_layers=2, hidden_dim=512):
    X_train, y_train, X_test, y_test = preprocess_data(X_test, y_test, input_dir)
    model = CoreModelFactory.create(
        model_type, input_dim=X_train.shape[1], class_num=class_num,
        num_hidden_layers=num_hidden_layers,
        hidden_dim=hidden_dim
    )
    trained_model = train_model(model, X_train, y_train, X_test, y_test, save_name, report, studentOrTeacher=studentOrTeacher)
    evaluate_model(trained_model, X_test, y_test, report, wandb_log=True, studentOrTeacher=studentOrTeacher)


# preprocess.py

def preprocess_data(X_test, y_test, input_dir):
    # {..private...}
    return X_train, y.to_numpy(), X_test, y_test


# train.py


def train_model(model, X_train, y_train, X_test, y_test, save_name, report, studentOrTeacher="student"):
    device = torch.device("cuda")
    model = model.to(device)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                            torch.tensor(y_train, dtype=torch.long).to(device)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                           torch.tensor(y_test, dtype=torch.long).to(device)), batch_size=64, shuffle=False)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(100):
        # for X_batch, y_batch in train_loader:
            # {..private...}

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save model and evaluate
    os.makedirs(os.path.dirname(save_name), exist_ok=True)  # Đảm bảo thư mục tồn tại
    torch.save(model.state_dict(), save_name)
    return model


# evaluate.py


def evaluate_model(model, X_test, y_test, report, wandb_log=False, prefix="", studentOrTeacher="student", downstream_model_name=""):
    # {..private...}


    # --- log bảng lên wandb ---
    if wandb_log:
        wandb_table = wandb.Table(
            columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
            data=table_data
        )
        wandb.log({f"Eval_{studentOrTeacher}_report": wandb_table})
        wandb.log({f"Eval_{studentOrTeacher}_acc": 100 * report_dict["accuracy"]})

    return report_dict["accuracy"]
