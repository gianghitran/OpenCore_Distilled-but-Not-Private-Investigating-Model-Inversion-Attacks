import torch
import torch.nn as nn

class MLP_General(nn.Module):
    def __init__(self, input_dim, class_num, num_hidden_layers="???", hidden_dim="???"):
        super().__init__()
        layers = []
        # Block đầu tiên: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        # Các block hidden: hidden_dim -> hidden_dim (num_hidden_layers lần)
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, class_num))
        self.model = nn.Sequential(*layers)
        self.num_hidden_layers = num_hidden_layers

    def forward(self, x):
        _, logits = self.extract_features_logits(x)
        return logits

    def extract_features_logits(self, x):
        for i in range(len(self.model) - 1):
            x = self.model[i](x)
        features = x
        logits = self.model[-1](features)
        return features, logits