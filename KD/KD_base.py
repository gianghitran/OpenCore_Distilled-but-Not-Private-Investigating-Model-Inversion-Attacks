import torch
import torch.nn as nn

class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.student_loss_fn = None # Initialize to None

    def forward(self, x):
        # {..private...}
        return student_pred, teacher_pred

   
    def train_step(self, x, y):
        self.train()
        # {..private...}
        return loss

    def evaluate(self, dataloader, device):
        self.eval()
        correct = 0
        total = 0
        total_loss = 0
        # with torch.no_grad():
            # {..private...}
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
