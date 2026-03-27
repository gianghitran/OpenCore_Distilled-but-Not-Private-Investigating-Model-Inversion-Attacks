import os
import pandas as pd
import torch
import numpy as np
import wandb


from utils.saveReport import save_report
from attack import AttackFactory
from model.core_model_factory import CoreModelFactory
from training.preprocessing import Preprocessing
from training.training import TrainingTeacher
from KD import KDFactory
from evaluation import qualityreport, KNNDist, attackUtility, attackAcc
from utils.plot import plot_tsne

def setup_and_preprocessing(cfg, orig_cwd):
    data_name = cfg.dataset.name
    data_csv = data_name + ".csv"
    trainset = os.path.join(orig_cwd, "data", f"train_set_{data_csv}")
    testset = os.path.join(orig_cwd, "data", f"test_set_{data_csv}")

    report_path = os.path.join(
        orig_cwd,
        f"Report_"
        f"dataset={cfg.dataset.name}"
        f"_model={cfg.model.type}"
        f"_layers={cfg.model.get('num_hidden_layers', 'NA')}"
        f"_hidden={cfg.model.get('hidden_dim', 'NA')}"
        f"_kd_method={cfg.kd.method}"
        f"_kd_alpha={cfg.kd.get('alpha', 'NA')}"
        f"_kd_temperature={cfg.kd.get('temperature', 'NA')}"
        f"_attack_method={cfg.attack.method}"
        f".csv"
    )

    if not (os.path.exists(trainset) and os.path.exists(testset)):
        print("Train/test set not found, preprocessing data...")
        raw_data_path = os.path.join(orig_cwd, "data", data_csv)
        preprocessor = Preprocessing(output_dir=os.path.join(orig_cwd, "data"))
        preprocessor.fit_transform(raw_data_path)

    train_df = pd.read_csv(trainset)
    test_df = pd.read_csv(testset)

    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values
    X_test = test_df.drop(columns=["label"]).values
    y_test = test_df["label"].values

    features = X_train.shape[1]
    label = len(pd.concat([train_df["label"], test_df["label"]]).unique())
    
    return X_train, y_train, X_test, y_test, features, label, report_path

def train_teacher_model(cfg, orig_cwd, X_train, y_train, X_test, y_test, features, label, report_path):
    data_name = cfg.dataset.name
    model_type = cfg.model.type
    teacher_model_path = os.path.join(
        orig_cwd, "model",
        f"Teacher_{data_name}_model={cfg.model.type}_layers={cfg.model.get('num_hidden_layers', 'NA')}_hidden={cfg.model.get('hidden_dim', 'NA')}.pkl"
    )
    if not os.path.exists(teacher_model_path):
        print("Teacher model not found, training...")
        os.makedirs(os.path.join(orig_cwd, "model"), exist_ok=True)
        trainer = TrainingTeacher(
            lambda in_dim, n_cls: CoreModelFactory.create(model_type, in_dim, n_cls),
            X_train, X_test, y_train, y_test, teacher_model_path, epochs=100, device="cuda", report=report_path
        )
        trainer.run()
    return teacher_model_path

def train_student_model(cfg, orig_cwd, X_train, y_train, X_test, y_test, features, label, teacher_model_path, report_path):
    data_name = cfg.dataset.name
    model_type = cfg.model.type
    student_model_path = os.path.join(
        orig_cwd, "model",
        f"Student_{data_name}_model={cfg.model.type}_layers={cfg.model.get('num_hidden_layers', 'NA')}_hidden={cfg.model.get('hidden_dim', 'NA')}.pkl"
    )
    if not os.path.exists(student_model_path):
        print("Student model not found, training...")
        teacher_model = CoreModelFactory.create(model_type, features, label)
        teacher_model.load_state_dict(torch.load(teacher_model_path))
        student_model = CoreModelFactory.create(model_type, features, label)
        trainer = KDFactory.get_trainer(cfg.kd.method)
        trainer(
            num_class=label,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            teacher=teacher_model,
            student=student_model,
            savepath=student_model_path,
            report=report_path,
            batch_size=64,
            epochs=100,
            alpha=cfg.kd.alpha,
            temperature=cfg.kd.temperature
        )
    return student_model_path

def perform_attack(cfg, orig_cwd, model_path, X_train, features, label, attack_type, data_name, report_path, studentOrTeacher="student"):
    print(f"Attacking {attack_type} model...")
    root_dir = os.path.join(orig_cwd, f"attack_{attack_type}")
    os.makedirs(root_dir, exist_ok=True)
    out_csv = os.path.join(root_dir, f"{attack_type}_attack_{data_name}.csv")

    attack = AttackFactory.create(cfg.attack.method, features, label, model_path, lambda in_dim, n_cls: CoreModelFactory.create(cfg.model.type, in_dim, n_cls), model_type=cfg.model.type)
    fea_mean = attack.compute_fea_mean(X_train)
    attack.attack(X_train, root_dir, out_csv, batch_size=32, fea_mean=fea_mean)
    
    X_attack, y_attack = None, None
    if os.path.exists(out_csv):
        df_attack = pd.read_csv(out_csv)
        X_attack = df_attack.drop(columns=["label"]).values
        y_attack = df_attack["label"].values

        qualityreport.qualityReport(X_train, features, data_name, out_csv, report_path, studentOrTeacher=studentOrTeacher)
        KNNDist.knn_distance(X_attack, X_train, report=report_path, device="cuda", batch_size=32, studentOrTeacher=studentOrTeacher)
    
    return out_csv, X_attack, y_attack

def run_evaluation(cfg, orig_cwd, X_test, y_test, label, attacked_data_path, evaluation_model_type, data_name, report_path):
    print(f"Training {evaluation_model_type} evaluation model...")
    eval_model_path = os.path.join(orig_cwd, "model", "modelpkl", f"{evaluation_model_type}_evaluation_model_{data_name}.pkl")
    os.makedirs(os.path.dirname(eval_model_path), exist_ok=True)
    save_report(report_path, [f"{evaluation_model_type.capitalize()} Evaluation Model"])
    attackUtility.evaluation(
        X_test, y_test, label, attacked_data_path, eval_model_path, report_path,
        model_type=cfg.model.type,
        studentOrTeacher=evaluation_model_type,
        num_hidden_layers=cfg.model.get("num_hidden_layers", 2),
        hidden_dim=cfg.model.get("hidden_dim", 512)
    )

def evaluate_downstream_models(orig_cwd, X_train, y_train, X_attack, y_attack, data_name, attack_type, report_path, prefix=""):
    print(f"Evaluating {attack_type} reconstructed data on downstream models...")
    save_report(report_path, [f"{attack_type} Attack Accuracy"])
    num_unique_y = len(set(y_attack))
    model_types = ["mlp4hidden", "mlp1hidden", "random_forest", "xgboost"]
    downstream_dir = os.path.join(orig_cwd, "downstream_model")
    os.makedirs(downstream_dir, exist_ok=True)

    for model_type in model_types:
        attackAcc.train_model(model_type, 100, downstream_dir, X_train, y_train)
        model_save_path = os.path.join(downstream_dir, f"{model_type}.pkl")
        acc = attackAcc.load_model_and_predict(model_type, model_save_path, X_attack, y_attack, num_unique_y)
        wandb.log({f"{prefix}Downstream/{model_type}": acc})
        save_report(report_path, [model_type, acc])

def generate_plots(orig_cwd, X_train, y_train, X_attack, y_attack, data_name, attack_type, cfg=None):
    print(f"Generating t-SNE plot for {attack_type} attack...")
    class_names = sorted(np.unique(np.concatenate([y_train, y_attack])))
    plot_dir = os.path.join(orig_cwd, "plot", data_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Thêm config vào tên file ảnh
    if cfg is not None:
        plot_path = os.path.join(
            plot_dir,
            f"tsne_{attack_type}_model={cfg.model.type}_layers={cfg.model.get('num_hidden_layers', 'NA')}_hidden={cfg.model.get('hidden_dim', 'NA')}_kd={cfg.kd.method}_attack={cfg.attack.method}.png"
        )
    else:
        plot_path = os.path.join(plot_dir, f"tsne_{attack_type}.png")

    plot_tsne(
        X_train, y_train,
        X_attack, y_attack,
        class_names,
        title=f"t-SNE Original vs {attack_type.capitalize()} Inverted - {data_name}",
        save_path=plot_path,
        prefix=f"{attack_type.capitalize()}/"
    )
