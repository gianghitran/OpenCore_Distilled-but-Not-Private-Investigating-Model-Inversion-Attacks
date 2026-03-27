import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, class_num):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear("???", "???"), nn.BatchNorm1d("???"), nn.ReLU(),
            nn.Linear("???", "???"), nn.BatchNorm1d("???"), nn.ReLU(),
            nn.Linear("???", "???")
        )
    def forward(self, x):
        _, logits = self.extract_features_logits(x)
        return logits

    def extract_features_logits(self, x):
        # Lấy output của lớp áp chót (trước Linear cuối cùng)
        for i in range(len(self.model) - 1):
            x = self.model[i](x)
        features = x
        logits = self.model[-1](features)
        return features, logits