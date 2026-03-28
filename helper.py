import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

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
    "Jailbreak": {"base_asr": 0.42},
    "Prompt Injection": {"base_asr": 0.61},
    "System Prompt Leakage": {"base_asr": 0.38},
}

def generate_harmbench_df(n_prompts=520):
    models = list(PUBLISHED_ASR.keys())
    attack_types = list(ATTACK_CATEGORIES.keys())
    rows = []

    for model, true_asr in PUBLISHED_ASR.items():
        for _ in range(n_prompts // len(models)):
            attack_type = np.random.choice(attack_types)
            attack_asr = (true_asr + ATTACK_CATEGORIES[attack_type]["base_asr"]) / 2
            success = int(np.random.rand() < attack_asr)

            rows.append({
                "model": model,
                "attack_type": attack_type,
                "attack_success": success,
                "prompt_length": np.random.randint(50, 500),
            })

    return pd.DataFrame(rows)

def compute_asr_by_model(df):
    return (
        df.groupby("model")["attack_success"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index(name="asr")
    )

def compute_heatmap_data(df):
    return (
        df.groupby(["model", "attack_type"])["attack_success"]
        .mean()
        .mul(100)
        .unstack()
        .fillna(0)
    )

def generate_epsilon_curve():
    eps = np.linspace(0, 0.2, 10)
    return pd.DataFrame({
        "epsilon": eps,
        "PGD Accuracy": np.clip(0.9 - eps * 3, 0, 1)
    })

def generate_per_class_vulnerability():
    classes = ["cat", "dog", "car", "truck"]
    return pd.DataFrame({
        "class": classes,
        "clean_accuracy": np.random.rand(len(classes)),
        "pgd_accuracy": np.random.rand(len(classes)),
    })

def generate_transferability_matrix():
    models = ["ResNet", "VGG", "MobileNet"]
    matrix = np.random.rand(3, 3) * 100
    return pd.DataFrame(matrix, index=models, columns=models)

def generate_jailbreak_attempts_over_time():
    days = pd.date_range("2024-01-01", periods=30)
    attempts = np.random.randint(50, 200, size=30)
    successes = (attempts * np.random.rand(30)).astype(int)

    return pd.DataFrame({
        "date": days,
        "attempts": attempts,
        "successes": successes,
        "asr": successes / attempts * 100
    })

def get_summary_stats(hb, eps):
    return {
        "overall_asr": round(hb["attack_success"].mean() * 100, 1),
        "pgd_fooling_at_eps3": 66.0,
        "models_tested": hb["model"].nunique(),
        "attack_categories": hb["attack_type"].nunique(),
        "total_attack_attempts": len(hb),
    }