import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_dim, class_num):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d("???", "???", kernel_size="???", padding="???"),
            nn.BatchNorm1d("???"),
            nn.ReLU(),
            nn.Conv1d("???", "???", kernel_size="???", padding="???"),
            nn.BatchNorm1d("???"),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear("???", "???"),
            nn.ReLU(),
            nn.Linear("???", "???")
        )
    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, input_dim]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def extract_features_logits(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        features = x.view(x.size(0), -1)
        logits = self.fc(features)
        return features, logits