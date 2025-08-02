from dotenv import load_dotenv
import librosa
import numpy as np
import os
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier

from utils.effects import *
from utils.model_eval import get_embed

load_dotenv()
emopia_dir = os.getenv("EMOPIA_DIR")
deam_dir = os.getenv("DEAM_DIR")
DATA_DIR = os.getenv("DATA_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")
SAMPLE_RATIO = float(os.getenv("SAMPLE_RATIO"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
NUM_LEVEL = int(os.getenv("NUM_LEVEL"))


def sample_dataset(df, dataset_name, sample_ratio=SAMPLE_RATIO, random_state=RANDOM_STATE):
    if dataset_name == 'deam':
        # simple random 20%
        return df.sample(frac=sample_ratio, random_state=random_state).reset_index(drop=True)

    elif dataset_name == 'emopia':
        # balanced 20%: total ~0.2*len, then even per class
        df = df.copy()
        n_target    = int(len(df) * sample_ratio)
        n_per_class = n_target // df['emo_class'].nunique()
        return (df.groupby('emo_class', group_keys=False)
              .apply(lambda grp: grp.sample(n=n_per_class, random_state=random_state))
              .reset_index(drop=True))

    else:
        return df


def extract_features_fx(df, processor, model, model_type, dataset_name, target_sr, max_duration=15):
    records = []

    effects = {
        "reverb": apply_reverb,
        "delay": apply_delay,
        "distortion": apply_distortion,
        "eq": apply_eq,
        "chorus": apply_chorus,
        "phaser": apply_phaser
    }

    print("Extracting Features:")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Features"):
        # 1. Load & preprocess audio
        audio, sr = librosa.load(row["path"], sr=None, mono=True)
        audio = audio[:sr * max_duration]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        for name, func in effects.items():
            for level in range(1, NUM_LEVEL+1):
                processed = func(audio, sr, level)
                embed = get_embed(processed, processor, model, model_type, target_sr)
                records.append({
                    "path": row["path"],
                    "features": embed,
                    "effect": name,
                    "level": level,
                })

    df_fx = pd.DataFrame(records)
    print(f"\nFeatures extracted")

    # Save df
    out_path = os.path.join(DATA_DIR, f"{dataset_name}_{model_type}_features_fx.pkl")
    save_df_fx = df_fx.copy()
    save_df_fx.to_pickle(out_path)
    print(f"Saved features to {out_path}")

    return df_fx


# ============== Regression Logic ==============

def load_regression_pipeline_fx(model_tag):
    # scaler
    with open(os.path.join(MODEL_DIR, f"{model_tag}_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # regressors
    reg_v = XGBRegressor()
    reg_v.load_model(os.path.join(MODEL_DIR, f"{model_tag}_valence.json"))

    reg_a = XGBRegressor()
    reg_a.load_model(os.path.join(MODEL_DIR, f"{model_tag}_arousal.json"))

    return scaler, reg_v, reg_a


def run_regression_pipeline_fx(df, model_tag, reg_valence, reg_arousal, scaler):
    X_raw = np.vstack(df["features"].values)
    X = scaler.transform(X_raw)

    values = {
        "valence": reg_valence,
        "arousal": reg_arousal
    }

    for name, model in values.items():
        df[name] = model.predict(X)
        print(f"Results extracted for {name}")

    # Save results Dataframe
    out_path = os.path.join(DATA_DIR, f"{model_tag}_results_fx.pkl")
    df.to_pickle(out_path)
    print(f"Saved results to {out_path}")


# ============== Classification Logic ==============

def load_classification_pipeline_fx(model_tag):
    # scaler
    with open(os.path.join(MODEL_DIR, f"{model_tag}_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # encoder
    with open(os.path.join(MODEL_DIR, f"{model_tag}_enc.pkl"), "rb") as f:
        encoder = pickle.load(f)

    # classifier
    clf = XGBClassifier()
    clf.load_model(os.path.join(MODEL_DIR, f"{model_tag}_clf.json"))

    return scaler, encoder, clf


def run_classification_pipeline_fx(df, model_tag, model, label_enc, scaler):
    X_raw = np.vstack(df["features"].values)
    X = scaler.transform(X_raw)

    preds = model.predict(X)
    print("Results Extracted")

    # Save results Dataframe
    df["results"] = label_enc.inverse_transform(preds)
    out_path = os.path.join(DATA_DIR, f"{model_tag}_results_fx.pkl")
    df.to_pickle(out_path)
    print(f"Saved results to {out_path}")


# ============== Pipeline Definition ==============

def run_pipeline_fx(dataset_name, dataset_dir, model_type, processor, model, target_sr):
    print(f"\n=== Running: {model_type.upper()} + {dataset_name.upper()} ===")

    # 1) load and sample dataset
    df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    df = sample_dataset(df, dataset_name)
    print(f"→ Sampled {len(df)} / {dataset_name}.csv (20%)")

    # 2) extract FX-augmented embeddings
    features_df = extract_features_fx(df, processor, model, model_type, dataset_name, target_sr)

    # 3) run the right downstream pipeline
    model_tag = f"{dataset_name}_{model_type}"
    if dataset_name == "deam":
        scaler, reg_v, reg_a = load_regression_pipeline_fx(model_tag)
        run_regression_pipeline_fx(features_df, model_tag, reg_v, reg_a, scaler)
    else:
        scaler, label_enc, clf = load_classification_pipeline_fx(model_tag)
        run_classification_pipeline_fx(features_df, model_tag, clf, label_enc, scaler)


def extract_results(mert, mert_proc, mert_sr, clap, clap_proc, clap_sr, qwen, qwen_proc, qwen_sr):
    run_pipeline_fx("deam", deam_dir, "mert", mert_proc, mert, mert_sr)
    run_pipeline_fx("deam", deam_dir, "clap", clap_proc, clap, clap_sr)
    run_pipeline_fx("deam", deam_dir, "qwen", qwen_proc, qwen, qwen_sr)
    run_pipeline_fx("emopia", emopia_dir, "mert", mert_proc, mert, mert_sr)
    run_pipeline_fx("emopia", emopia_dir, "clap", clap_proc, clap, clap_sr)
    run_pipeline_fx("emopia", emopia_dir, "qwen", qwen_proc, qwen, qwen_sr)
    print("Results extraction complete!\n")