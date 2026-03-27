import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

class BaseAttack:
    def __init__(self, feature_num, class_num, target_pkl, loss_fn, model_class, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.model = model_class(feature_num, class_num).to(self.device).eval()
        assert os.path.exists(target_pkl)
        self.model.load_state_dict(torch.load(target_pkl, map_location=self.device), strict=False)
        self.feature_num = feature_num
        self.class_num = class_num
        self.loss_fn = loss_fn

    # def process(self, x):
        # {..private...}
        

    def _call_loss_fn(self, noise, target_labels, fea_mean, lam):
        return self.loss_fn(self.model, noise, target_labels, fea_mean, lam)

    def attack_batch(
        # self, batch_size, target_labels,
        # alpha=???, beta=???, gama=???,
        # learning_rate=???, momentum=???,
        # wandb_prefix="", fea_mean=???, lam=???
    ):
        # {..private...}
        return "???"

    def attack(
        # self, X_train, root_dir, output_csv, batch_size=32,
        # alpha=???, beta=???, gama=???,
        # learning_rate=???, momentum=???,
        # fea_mean=???, lam=???
    ):        
        # {..private...}       
                attacked_data_np = attacked_data.cpu().numpy()
                target_class_np = target_class.cpu().numpy().reshape(-1, 1)
                attacked_data_with_label = np.hstack((attacked_data_np, target_class_np))
                df = pd.DataFrame(attacked_data_with_label, columns=columns)
                df.to_csv(output_csv, mode="a", header=False, index=False)
        # print(f"Attack results saved to {output_csv}")