import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb

from utils.dataloader import create_dataloader
from utils.saveReport import save_report

class Training:
    def __init__(self, model, optimizer, loss_fn, epochs=100, device="cuda", report=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.report = report

    def train(self, train_loader, val_loader=None):
        self.model.train()
        best_acc = -1
        best_state = None
        for epoch in range(self.epochs):
            # {..private...}
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}')
            wandb.log({"Training/Epoch": epoch + 1, "Loss": avg_loss})

            # Đánh giá trên validation/test set nếu có
            # if val_loader is not None:
                # {..private...}
            # else:
                # Nếu không có val_loader, lưu model cuối cùng
                # best_state = self.model.state_dict()

        # Sau khi train xong, trả về best_state
        return best_state, best_acc

    def evaluate(self, test_loader, log_wandb=True):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        if log_wandb:
            wandb.log({"Training/Teacher Accuracy": accuracy})
        return accuracy


class TrainingTeacher:
    def __init__(self, model_class, X_train, X_test, y_train, y_test, savepath, epochs=100, device="cuda", report=None):
        input_dim = X_train.shape[1]
        num_classes = len(set(y_train))
        self.model = model_class(input_dim, num_classes)
        self.trainer = Training("???")
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.savepath = savepath

    def run(self):
        train_loader = create_dataloader(self.X_train, self.y_train)
        test_loader = create_dataloader(self.X_test, self.y_test, shuffle=False)
        best_state, best_acc = self.trainer.train(train_loader, test_loader)
        os.makedirs(os.path.dirname(self.savepath), exist_ok=True)
        torch.save(best_state, self.savepath)
        print(f"Teacher model saved to {self.savepath}")
        print(f"Accuracy on test set: {best_acc}%")
        wandb.log({"Training/Teacher Accuracy": best_acc})
        
        if self.trainer.report is not None:
            save_report(self.trainer.report, ["Teacher Accuracy", best_acc])
            
        return self.model, best_acc, self.savepath

class TrainingDownstreamModel:
    def __init__(self, model_class, X_train, X_test, y_train, y_test, savepath, epochs=100, device="cuda"):
        input_dim = X_train.shape[1]
        num_classes = len(set(y_train))
        self.model = model_class(input_dim, num_classes)
        self.trainer = Training("???")
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.savepath = savepath

    def run(self):
        train_loader = create_dataloader(self.X_train, self.y_train)
        test_loader = create_dataloader(self.X_test, self.y_test, shuffle=False)
        self.trainer.train(train_loader)
        acc = self.trainer.evaluate(test_loader)
        os.makedirs(os.path.dirname(self.savepath), exist_ok=True)
        torch.save(self.model.state_dict(), self.savepath)
        print(f"Downstream model saved to {self.savepath}")
        return self.model, acc, self.savepath
