import torch
import torch.nn.functional as F
from attack.baseAttack import BaseAttack

# def improved_loss(attack_instance, noise, target_labels, fea_mean, lam=0.1):
    # {..private...}
    

class ImprovedAttack(BaseAttack):
    def __init__(self, feature_num, class_num, target_pkl, model_class, device=None, model_type="MLP"):
        self.model_type = model_type
        super().__init__(feature_num, class_num, target_pkl, improved_loss, model_class, device)

    # def extract_features_logits(self, x):
        # {..private...}


    # def compute_fea_mean(self, X_train):
        # {..private...}


    # def _call_loss_fn("???"):
        # {..private...}
        