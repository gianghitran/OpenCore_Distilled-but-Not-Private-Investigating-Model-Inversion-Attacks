import os
import shutil
from datetime import date
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from utils.saveReport import save_report
from utils.pipeline_steps import (
    setup_and_preprocessing,
    train_teacher_model,
    train_student_model,
    perform_attack,
    run_evaluation,
    evaluate_downstream_models,
    generate_plots
)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_name = cfg.dataset.name
    print(f"================================== Running dataset: {data_name} ==================================")

    outputs_dir = os.path.join(os.getcwd(), "outputs")

    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir, ignore_errors=True)

    wandb.init(
        project="Data-Leakage-Evaluation-of-Knowledge-Distillation-via-Model-Inversion-Attack",
        name=f"{cfg.dataset.name}_{cfg.kd.method}_{cfg.attack.method}",
        config=OmegaConf.to_container(cfg, resolve=True), 
        reinit=True
    )

    orig_cwd = hydra.utils.get_original_cwd()
    
    today = date.today()
    formatted_date = today.strftime("%d/%m/%Y")
    
    X_train, y_train, X_test, y_test, features, label, report_path = setup_and_preprocessing(cfg, orig_cwd)
    save_report(report_path, [formatted_date])

    teacher_model_path = train_teacher_model(cfg, orig_cwd, X_train, y_train, X_test, y_test, features, label, report_path)
    student_model_path = train_student_model(cfg, orig_cwd, X_train, y_train, X_test, y_test, features, label, teacher_model_path, report_path)

    # Teacher Attack
    teacher_attack_data_path, X_teacher_attack, y_teacher_attack = perform_attack(cfg, orig_cwd, teacher_model_path, X_train, features, label, "teacher", data_name, report_path, studentOrTeacher="teacher")

    # Student Attack
    student_attack_data_path, X_student_attack, y_student_attack = perform_attack(cfg, orig_cwd, student_model_path, X_train, features, label, "student", data_name, report_path, studentOrTeacher="student")

    # Evaluation
    run_evaluation(cfg, orig_cwd, X_test, y_test, label, teacher_attack_data_path, "teacher", data_name, report_path)
    run_evaluation(cfg, orig_cwd, X_test, y_test, label, student_attack_data_path, "student", data_name, report_path)

    # Downstream Model Evaluation
    evaluate_downstream_models(orig_cwd, X_train, y_train, X_teacher_attack, y_teacher_attack, data_name, "Teacher", report_path, prefix="Teacher/")
    evaluate_downstream_models(orig_cwd, X_train, y_train, X_student_attack, y_student_attack, data_name, "Student", report_path, prefix="Student/")

    # Plotting
    generate_plots(orig_cwd, X_train, y_train, X_teacher_attack, y_teacher_attack, data_name, "teacher", cfg)
    generate_plots(orig_cwd, X_train, y_train, X_student_attack, y_student_attack, data_name, "student", cfg)

    wandb.finish() 

if __name__ == '__main__':
    main()
