import torch
from tqdm import tqdm
from utils.saveReport import save_report
import wandb
import numpy as np

def knn_distance(reconstructed, private, report=None, device="cuda", batch_size=512, studentOrTeacher="student"):
    """
    Compute average minimum Euclidean distance from reconstructed samples
    to the closest real sample (tabular version).

    Args:
        reconstructed: torch.Tensor or np.ndarray [N, D] - reconstructed samples
        private: torch.Tensor or np.ndarray [M, D] - real private training samples
        report: Optional object for saving report
        device: str - device to run on
        batch_size: int - number of reconstructed samples to process per batch
        studentOrTeacher: str - model type ("student" or "teacher")

    Returns:
        float - average KNN distance
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(reconstructed, (np.ndarray,)):
        reconstructed = torch.tensor(reconstructed, dtype=torch.float32)
    if isinstance(private, (np.ndarray,)):
        private = torch.tensor(private, dtype=torch.float32)

    reconstructed = reconstructed.to(device)
    private = private.to(device)
    min_dists = []

    with torch.no_grad():
        num_reconstructed = reconstructed.size(0)
        for i in tqdm(range(0, num_reconstructed, batch_size), desc="KNN Batch"):
            # {..private...}

            min_dists.append(min_batch_dist.cpu())

    all_min_dists = torch.cat(min_dists)
    
    knn_dist_score = all_min_dists.mean().item()
    
    print(f"KNN Distance : {knn_dist_score:.4f}")
    wandb.log({f"Evaluation_{studentOrTeacher}/KNN Distance": knn_dist_score}) # attack/KnnDistance

    if report is not None:
        save_report(report, ["KNN Feature Distance", f"{knn_dist_score:.4f}"])
    
    return knn_dist_score
