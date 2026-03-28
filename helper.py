import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
CIFAR10_CLEAN_ACCURACY = 0.918

PUBLISHED_ASR = {
    "Llama-2-7B": 0.08,
    "Llama-2-13B": 0.11,
    "Vicuna-7B": 0.68,
    "Vicuna-13B": 0.73,
    "Mistral-7B": 0.55,
    "GPT-3.5-turbo": 0.31,
    "GPT-4": 0.09,
    "Claude-2": 0.04,
}

ATTACK_CATEGORIES = {
    "Jailbreak": {"base_asr": 0.42, "color": "#f97316"},
    "Prompt Injection": {"base_asr": 0.61, "color": "#ef4444"},
    "System Prompt Leakage": {"base_asr": 0.38, "color": "#22d3ee"},
}

def generate_harmbench_df(n_prompts=520):
    models = list(PUBLISHED_ASR.keys())
    attack_types = list(ATTACK_CATEGORIES.keys())
    rows = []

    n_per_model = max(1, n_prompts // len(models))

    for model, true_asr in PUBLISHED_ASR.items():
        for _ in range(n_per_model):
            attack_type = RNG.choice(attack_types)
            attack_asr = (true_asr + ATTACK_CATEGORIES[attack_type]["base_asr"]) / 2
            success = int(RNG.random() < attack_asr)

            rows.append({
                "model": model,
                "attack_type": attack_type,
                "attack_success": success,
                "prompt_length": int(RNG.integers(50, 500)),
            })

    return pd.DataFrame(rows)

def compute_asr_by_model(df):
    result = (
        df.groupby("model")["attack_success"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index(name="asr")
    )
    return result.sort_values("asr", ascending=True).reset_index(drop=True)

def compute_asr_by_attack_type(df):
    result = (
        df.groupby("attack_type")["attack_success"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index(name="asr")
    )
    return result.sort_values("asr", ascending=False).reset_index(drop=True)

def compute_heatmap_data(df):
    return (
        df.groupby(["model", "attack_type"])["attack_success"]
        .mean()
        .mul(100)
        .unstack()
        .reindex(index=list(PUBLISHED_ASR.keys()), columns=list(ATTACK_CATEGORIES.keys()))
        .fillna(0)
    )

def generate_epsilon_curve():
    eps = np.array([0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20])

    fgsm_accuracy = np.array([0.918, 0.83, 0.72, 0.58, 0.41, 0.22, 0.15, 0.10, 0.07, 0.05])
    pgd_accuracy = np.array([0.918, 0.74, 0.56, 0.34, 0.19, 0.06, 0.03, 0.02, 0.01, 0.01])

    fgsm_fooling = np.round((1 - fgsm_accuracy) * 100, 1)
    pgd_fooling = np.round((1 - pgd_accuracy) * 100, 1)

    return pd.DataFrame({
        "epsilon": eps,
        "FGSM Accuracy": fgsm_accuracy,
        "PGD Accuracy": pgd_accuracy,
        "FGSM Fooling Rate": fgsm_fooling,
        "PGD Fooling Rate": pgd_fooling,
    })

def generate_per_class_vulnerability():
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    clean_accuracy = np.array([0.94, 0.95, 0.90, 0.86, 0.91, 0.87, 0.95, 0.91, 0.94, 0.93])
    pgd_accuracy = np.array([0.42, 0.39, 0.27, 0.18, 0.25, 0.20, 0.34, 0.31, 0.41, 0.37])

    df = pd.DataFrame({
        "class": classes,
        "clean_accuracy": clean_accuracy,
        "pgd_accuracy": pgd_accuracy,
    })
    df["accuracy_drop"] = ((df["clean_accuracy"] - df["pgd_accuracy"]) * 100).round(1)
    return df.sort_values("accuracy_drop", ascending=False).reset_index(drop=True)

def generate_transferability_matrix():
    models = ["ResNet-18", "VGG-16", "MobileNetV2", "DenseNet-121"]
    matrix = np.array([
        [88, 41, 36, 29],
        [39, 85, 31, 27],
        [35, 33, 83, 25],
        [31, 28, 24, 81],
    ])
    return pd.DataFrame(matrix, index=models, columns=models)

def generate_jailbreak_attempts_over_time(periods=60):
    days = pd.date_range("2024-01-01", periods=periods)

    trend = np.linspace(90, 160, periods)
    seasonality = 18 * np.sin(np.linspace(0, 3.5 * np.pi, periods))
    noise = RNG.normal(0, 9, periods)
    attempts = np.maximum(45, (trend + seasonality + noise)).astype(int)

    asr_ratio = np.clip(0.35 + 0.10 * np.sin(np.linspace(0, 2.5 * np.pi, periods)) + RNG.normal(0, 0.03, periods), 0.18, 0.70)
    successes = np.minimum(attempts, np.round(attempts * asr_ratio).astype(int))

    return pd.DataFrame({
        "date": days,
        "attempts": attempts,
        "successes": successes,
        "asr": np.round(successes / attempts * 100, 1)
    })

def get_summary_stats(hb, eps):
    asr_model = compute_asr_by_model(hb)
    asr_attack = compute_asr_by_attack_type(hb)
    pgd_eps3 = float(eps.loc[np.isclose(eps["epsilon"], 0.03), "PGD Fooling Rate"].iloc[0])

    return {
        "overall_asr": round(hb["attack_success"].mean() * 100, 1),
        "pgd_fooling_at_eps3": round(pgd_eps3, 1),
        "most_vulnerable_model": asr_model.iloc[-1]["model"],
        "most_dangerous_attack": asr_attack.iloc[0]["attack_type"],
        "models_tested": hb["model"].nunique(),
        "attack_categories": hb["attack_type"].nunique(),
        "total_attack_attempts": len(hb),
    }