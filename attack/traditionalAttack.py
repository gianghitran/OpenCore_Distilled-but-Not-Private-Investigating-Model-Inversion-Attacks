import torch
import torch.nn as nn
from attack.baseAttack import BaseAttack

# def traditional_loss(model, noise, target_labels, fea_mean=None, lam=None):
    # {..private...}
    

class TraditionalAttack(BaseAttack):
    def __init__(self, feature_num, class_num, target_pkl, model_class, device=None, **kwargs):
        super().__init__(feature_num, class_num, target_pkl, traditional_loss, model_class, device)

    # def compute_fea_mean(self, X_train):
        # {..private...}
