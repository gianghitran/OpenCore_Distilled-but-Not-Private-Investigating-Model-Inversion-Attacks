import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_planes, planes),
            nn.BatchNorm1d(planes),
            nn.ReLU(),
            nn.Linear(planes, planes),
            nn.BatchNorm1d(planes)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        out += x
        return self.relu(out)

class ResNetTabular(nn.Module):
    def __init__(self, input_dim, class_num):
        super().__init__()
        self.fc1 = nn.Linear("???", "???")
        self.block1 = BasicBlock("???", "???")
        self.block2 = BasicBlock("???", "???")
        self.fc2 = nn.Linear("???", "???")

    def forward(self, x):
        _, logits = self.extract_features_logits(x)
        return logits

    def extract_features_logits(self, x):
        out = self.fc1(x)
        out = self.block1(out)
        features = self.block2(out)
        logits = self.fc2(features)
        return features, logits