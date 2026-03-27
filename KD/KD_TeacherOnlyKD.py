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

class TeacherOnlyDistiller(Distiller):
    def __init__(self, student, teacher):
        super(TeacherOnlyDistiller, self).__init__(student, teacher)

    def compile(
        self,
        optimizer,
        metrics,
        temperature=3
    ):
        self.optimizer = optimizer
        self.metrics = metrics
        # self.distillation_loss_fn = ???
        self.temperature = temperature

    def compute_loss(self, x, y):
       # {..private...}
        return loss

def train_student_teacher_only(num_class, X_train, y_train, X_test, y_test, teacher, student, savepath, report, batch_size=64, epochs=100, alpha=0.1, temperature=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student.to(device)

    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    accuracy_metric = torchmetrics.Accuracy(num_classes=num_class, task='multiclass').to(device)

    distiller = TeacherOnlyDistiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=optim.Adam(student.parameters()),
        metrics=[accuracy_metric],
        temperature=temperature,
    )

    for epoch in range(epochs):
        # for X_batch, y_batch in train_loader:
            # {..private...}
        print(f"Epoch {epoch+1}, Distillation Loss: {loss.item()}")
        wandb.log({"KD/Student Epoch - TeacherOnlyKD": epoch + 1, "Student Loss - TeacherOnlyKD": loss.item()})

    avg_loss, accuracy = distiller.evaluate(test_loader, device)
    print(f"Accuracy of the student model on the test set: {accuracy}%")
    wandb.log({"KD/Student Accuracy - TeacherOnlyKD": accuracy})
    save_report(report, ["Student Accuracy - TeacherOnlyKD", f"{accuracy}%"])

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    torch.save(distiller.student.state_dict(), savepath)
    print(f"Student model saved to {savepath}")

    return distiller.student, accuracy
