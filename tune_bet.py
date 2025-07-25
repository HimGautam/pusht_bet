import subprocess
import re
import optuna
import optuna.visualization.matplotlib as vis_matplotlib

# Fixed part of your train CLI:
BASE_CMD = [
    "python", "-m", "lerobot.scripts.train",
    "--policy.type", "behavior_transformer",
    "--dataset.repo_id", "lerobot/pusht_image",
    "--env.type", "pusht",
    "--env.obs_type", "pixels_agent_pos",
    "--steps", "20000",
    "--log_freq", "200",
    "--save_freq", "1000",
    "--eval_freq", "1000",
    "--eval.n_episodes", "10",
    "--eval.batch_size", "10",
    "--policy.push_to_hub", "false",
    "--wandb.enable", "true",
    "--wandb.project", "pusht_bet",
    "--batch_size", "64",
]

# weight on success rate when combining
ALPHA = 1.0

def objective(trial):
    # sample a few important hyper-parameters
    weight_decay   = trial.suggest_float("optimizer_weight_decay", 0.01,   0.2,  log=True)
    dropout        = trial.suggest_float("dropout_rate",           0.0,    0.4)
    lambda_offset  = trial.suggest_float("lambda_offset",          5,   15)

    # each trial writes its own output dir
    outdir = f"./bet_run/trial_{trial.number}"
    cmd = BASE_CMD + [
        "--output_dir", outdir,
        "--policy.optimizer_weight_decay",  str(weight_decay),
        "--policy.dropout_rate",            str(dropout),
        "--policy.lambda_offset",           str(lambda_offset),
    ]

    # run and capture logs
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed:\n{completed.stdout}")

    # parse metrics
    sum_reward  = None
    success_rate = None
    for line in completed.stdout.splitlines():
        m_r = re.search(r"∑rwrd[:=]\s*([0-9.+-eE]+)", line)
        if m_r:
            sum_reward = float(m_r.group(1))
        m_s = re.search(r"success[:=]\s*([0-9.+-eE]+)", line)
        if m_s:
            success_rate = float(m_s.group(1))

    if sum_reward is None or success_rate is None:
        trailer = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(
            "Could not parse ∑rwrd or success from output. "
            "Last 20 lines:\n" + trailer
        )

    # combine reward + alpha * success_rate into one objective
    combined_score = sum_reward + ALPHA * success_rate

    # store success for later inspection
    trial.set_user_attr("success_rate", success_rate)

    # print both metrics and the combined score
    print(
        f"[Trial {trial.number}] "
        f"∑rwrd = {sum_reward:.3f}, "
        f"success = {success_rate:.3f}, "
        f"combined = {combined_score:.3f}"
    )

    return combined_score

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="bet_optuna",
        storage="sqlite:///bet_optuna.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1000),
    )
    study.optimize(objective, n_trials=9, show_progress_bar=True)

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df.to_csv("bet_optuna_results.csv", index=False)

    best = study.best_trial
    print("=== Best Trial ===")
    print(f"∑rwrd:    {best.value - ALPHA * best.user_attrs['success_rate']:.3f}")
    print(f"success:  {best.user_attrs['success_rate']:.3f}")
    print(f"combined: {best.value:.3f}")

    fig1 = vis_matplotlib.plot_optimization_history(study)
    fig1.figure.savefig("optuna_history.png")
    fig2 = vis_matplotlib.plot_param_importances(study)
    fig2.figure.savefig("optuna_param_importance.png")