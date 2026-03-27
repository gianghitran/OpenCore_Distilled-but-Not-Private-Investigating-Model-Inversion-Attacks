# OpenCore_Distilled-but-Not-Private-Investigating-Model-Inversion-Attacks
This repository contains the open-core implementation and experimental framework for the paper: 'Distilled but Not Private: Investigating Model Inversion Attacks on IoT Intrusion Detection Systems.' It explores how the knowledge distillation (KD) technique impacts training-data leakage.
## ⚠️ Disclaimer & Proprietary Notice
> Please note that this repository contains the **Open Core** version of the research framework. Due to the sensitive nature of security research and intellectual property (IP) agreements, certain critical modules and proprietary algorithms have been **selectively redacted or omitted**. 
> 
> The provided code is intended for academic demonstration and reproducibility of the core findings, rather than a full-scale production deployment.


## Project Overview
This project evaluates the data privacy risks of Knowledge Distillation (KD) in IoT-based Network Intrusion Detection Systems (NIDS). We utilize Model Inversion Attacks (MIA) to demonstrate that sensitive training data can be reconstructed from distilled student models, even when the teacher model remains protected.

### Under this setting, we investigate the following research questions (RQs):
- RQ1: How effective are MIA methods against different MLP models?
- RQ2: How effective are MIA methods against different KD methods?
- RQ3: What is the impact of teacher--student architectures on MIA effectiveness?


### Key Contributions
- Unified R&D Framework: Architected a modular system using Software Design Patterns to decouple training, distillation, and attack logic. This ensures high extensibility for integrating new architectures or threat models.
- Scalable Configuration Management (Hydra): Integrated Hydra to manage complex hierarchical configurations. This enables seamless switching between datasets, model variants, and attack strategies via CLI, and leverages Hydra Sweepers for automated, large-scale hyperparameter optimization.
- End-to-End MLOps Pipeline (W&B): Systematized experiment tracking using Weights & Biases. This allows for real-time monitoring of attack convergence, logging of reconstruction fidelity, and visualization of high-dimensional feature distributions across hundreds of runs.
- MIA Benchmarking: Conducted a comprehensive comparative study between Posterior-based and Logit-based inversion methods across 4 public tabular IDS datasets, providing a robust security baseline for IoT NIDS.
- Privacy Insights: Delivered empirical evidence that ground-truth supervision in traditional KD is a primary driver of data leakage, advocating for the adoption of privacy-aware distillation objectives.

## System Architecture
The repository is structured for high extensibility and reproducibility:
> - KD/: Logic for Knowledge Distillation variants (Traditional KD, Teacher-Only KD).
> - attack/: Optimization-based Model Inversion methods (Traditional, Improved).
> - configs/: Centralized experiment management using Hydra YAML files.
> - model/: Architecture definitions
> - training/: Pipelines for Teacher and Student model training.
> - evaluation/: Metrics computation (Attack Accuracy, KNN Distance, Downstream Utility).
> - utils/: Data Loader, Tracker, Report processing.

## Getting Started
### 1. Installation
```Bash
pip install -r requirements.txt
wandb login
```
### 2. Experiment Management with Hydra
Switch between models, datasets, and attack methods seamlessly via CLI without modifying the source code.
- Run a single experiment:

```Bash
# Execute with default parameters from config.yaml
python main.py

# Override configurations for specific scenarios
python main.py model.type=ResNet dataset.name=cicids2017 kd.temperature=5
```

- Run large-scale benchmarks (Multirun):
Utilize the built-in Hydra Sweeper to automatically iterate through all combinations of models, KD methods, and attacks:

```Bash
python main.py --multirun
```

## Experiment Tracking (W&B)
All experiments are logged to Weights & Biases. We monitor: Attack convergence (Inversion Loss), Reconstruction fidelity (High-dimensional feature distributions), Statistical benchmarking across multiple privacy metrics.
...

## Citation
```Code snippet
@inproceedings{tran2026distillednprivate,
  title={Distilled but Not Private: Investigating Model Inversion Attacks on IoT Intrusion Detection Systems},
  author={Tran, Gia-Nghi and Nguyen, Da-Vit and Nguyen, Dat-Thinh and Le-Khac, Nhien-An and Le, Kim-Hung},
  booktitle={---} (submited),
  year={2026}
}
```
