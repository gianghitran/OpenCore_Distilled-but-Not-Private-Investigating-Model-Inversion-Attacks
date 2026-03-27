import os
import torch
import pandas as pd
import numpy as np
import wandb
import logging
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from utils.saveReport import save_report

logging.getLogger("sdv").setLevel(logging.WARNING)

def qualityReport(X_train, feature_num, name, csv_filename, report, studentOrTeacher="student"):

    device = torch.device("cuda")
    
    def prepare_real_data(X, feature_dim):
        print(f"[Info] Feature count in X_scaled: {X.shape[1]}")
        print(f"[Info] Feature dimension provided: {feature_dim}")
        return pd.DataFrame(X, columns=[f'feature_{j}' for j in range(feature_dim)])

    def load_synthetic_data(csv_path, feature_dim):
        df = pd.read_csv(csv_path)
        df = df.iloc[:, :-1]  # Remove label column if present
        df = df.astype(np.float64)
        return pd.DataFrame(df, columns=[f'feature_{j}' for j in range(feature_dim)])

    # def evaluate_sdv_quality(real_df, synth_df):
        # {..private...}


    def extract_scores(quality_report):
        col_shape = quality_report.get_details(property_name="Column Shapes")
        col_pair = quality_report.get_details(property_name="Column Pair Trends")
        
        # def score_from_detail(detail):
                # {..private...}


        return {
            "col_shape_score": score_from_detail(col_shape),
            "col_pair_score": score_from_detail(col_pair),
            "overall_score": quality_report.get_score()
        }

    def log_scores_wandb(scores_dict, prefix):
        wandb.log({
            f"Evaluation_{studentOrTeacher}/qualityReport/{prefix} Column Shapes Score": scores_dict["col_shape_score"],
            f"Evaluation_{studentOrTeacher}/qualityReport/{prefix} Column Pair Trends Score": scores_dict["col_pair_score"],
            f"Evaluation_{studentOrTeacher}/qualityReport/{prefix} Overall Score": scores_dict["overall_score"]
        })

    def save_report_scores(report, scores_dict, model_type):
        save_report(report, [f"Quality Report {model_type}"])
        save_report(report, ["Column Shapes Score", f"{scores_dict['col_shape_score']:.2%}"])
        save_report(report, ["Column Pair Trends Score", f"{scores_dict['col_pair_score']:.2%}"])
        save_report(report, ["Overall Score (Average)", f"{scores_dict['overall_score']:.2%}"])

    real_df = prepare_real_data(X_train, feature_num)
    synthetic_df = load_synthetic_data(csv_filename, feature_num)

    quality_report = evaluate_sdv_quality(real_df, synthetic_df)
    print(quality_report)

    scores = extract_scores(quality_report)
    log_scores_wandb(scores, name)

    model_type = os.path.basename(csv_filename).split("_")[0].capitalize()
    save_report_scores(report, scores, model_type)

    # Optional MSE calculation (tensor format) — not used here, but retained if needed
