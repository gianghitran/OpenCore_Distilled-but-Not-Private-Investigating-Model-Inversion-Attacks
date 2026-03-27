import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torchmetrics

from .KD_base import Distiller
from utils.dataloader import create_dataloader
from utils.saveReport import save_report

class TraditionalDistiller(Distiller):
    def __init__(self, student, teacher):
        super(TraditionalDistiller, self).__init__(student, teacher)

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        # alpha=???,
        # temperature=???
    ):
        self.optimizer = optimizer
        self.metrics = metrics
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, x, y):
       # {..private...}
        return loss

def train_student_traditionally(num_class, X_train, y_train, X_test, y_test, teacher, student, savepath, report, batch_size=64, epochs=100, alpha=0.1, temperature=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student.to(device)

    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    accuracy_metric = torchmetrics.Accuracy(num_classes=num_class, task='multiclass').to(device)

    distiller = TraditionalDistiller(student=student, teacher=teacher)
    distiller.compile(
        # optimizer=???
        # metrics=???
        # student_loss_fn=???
        # distillation_loss_fn=???
        # alpha=alpha,
        # temperature=temperature,
    )

    for epoch in range(epochs):
            # {..private...}
        print(f"Epoch {epoch+1}, Distillation Loss: {loss.item()}")
        wandb.log({"KD/Student Epoch - Traditional": epoch + 1, "Student Loss - Traditional": loss.item()})

    avg_loss, accuracy = distiller.evaluate(test_loader, device)
    print(f"Accuracy of the student model on the test set: {accuracy}%")
    wandb.log({"KD/Student Accuracy - Traditional": accuracy})
    save_report(report, ["Student Accuracy - Traditional", f"{accuracy}%"])

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    torch.save(distiller.student.state_dict(), savepath)
    print(f"Student model saved to {savepath}")

    return distiller.student, accuracy
