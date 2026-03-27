import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb
from model.model_factory import ModelFactory

MODEL_LOADERS = {
    "mlp4hidden": lambda path, input_dim, class_num, device: ModelFactory.create("mlp4hidden", input_dim, class_num).model,
    "mlp1hidden": lambda path, input_dim, class_num, device: ModelFactory.create("mlp1hidden", input_dim, class_num).model,
    "random_forest": lambda path, *_: joblib.load(path),
    "xgboost": lambda path, *_: joblib.load(path),
}

MODEL_PREDICTORS = {
    "mlp4hidden": lambda model, X, device: model(torch.tensor(X, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy(),
    "mlp1hidden": lambda model, X, device: model(torch.tensor(X, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy(),
    "random_forest": lambda model, X, device=None: model.predict(X),
    "xgboost": lambda model, X, device=None: model.predict(X),
}

def train_model(model_type,epochs,model_path, X, y):
    input_dim = X.shape[1]
    class_num = len(set(y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if "mlp" in model_type:
        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

        model_wrapper = ModelFactory.create(model_type, input_dim, class_num)
        model = model_wrapper.model  # access raw nn.Module
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        best_acc = -1
        best_state = None
        for epoch in range(epochs):  # số epochs
            # {..private...}

            model.train()
        # Save best model
        model_save_path = os.path.join(model_path, f"{model_type}.pkl")
        torch.save(best_state, model_save_path)

    elif model_type == "random_forest":
        # {..private...}

        model_save_path = os.path.join(model_path, "random_forest.pkl")
        joblib.dump(model, model_save_path)

    elif model_type == "xgboost":
        # {..private...}

        model_save_path = os.path.join(model_path, "xgboost.pkl")
        joblib.dump(model, model_save_path)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"{model_type} trained and saved to {model_path}")



# Cách dùng ======================================

# attack_accuracy("mlp4hidden",100, X, y)
# attack_accuracy("random_forest",100, X, y)


def load_model_and_predict(model_type, model_path, X, y, class_num, studentOrTeacher="teacher"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    loader = MODEL_LOADERS.get(model_type)
    predictor = MODEL_PREDICTORS.get(model_type)
    if loader is None or predictor is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = loader(model_path, input_dim, class_num, device)
    if "mlp" in model_type:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        with torch.no_grad():
            preds = predictor(model, X, device)
    else:
        preds = predictor(model, X)
    acc = accuracy_score(y, preds)

    wandb.log({f"Evaluation_{studentOrTeacher}/Downstream_{model_type}_Accuracy": acc})
    return acc
