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
wtf_dir = os.getenv("WITHEFLOW_DIR")
DATA_DIR = os.getenv("DATA_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")

SAMPLE_RATIO = float(os.getenv("SAMPLE_RATIO"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
NUM_LEVEL = int(os.getenv("NUM_LEVEL"))

# ======================== Band Scenarios ========================
EFFECT_ORDER = ["distortion", "eq", "reverb", "delay", "chorus", "phaser"]

SCENARIOS = {
    "Original": {},
    "Pink Floyd": {"distortion": 3, "eq": 6, "reverb": 9, "delay": 7, "chorus": 2, "phaser": 4},
    "Rage Against The Machine": {"distortion": 9, "eq": 5, "reverb": 1, "delay": 2, "chorus": 6, "phaser": 1},
    "U2": {"distortion": 4, "eq": 7, "reverb": 8, "delay": 10, "chorus": 2, "phaser": 3},
}

def apply_effect_by_name(audio, sr, effect_name, level):
    if effect_name == "reverb":
        return apply_reverb(audio, sr, level)
    if effect_name == "delay":
        return apply_delay(audio, sr, level)
    if effect_name == "distortion":
        return apply_distortion(audio, sr, level)
    if effect_name == "eq":
        return apply_eq(audio, sr, level)
    if effect_name == "chorus":
        return apply_chorus(audio, sr, level)
    if effect_name == "phaser":
        return apply_phaser(audio, sr, level)
    raise ValueError(f"Unknown effect '{effect_name}'")

def apply_scenario_chain(audio, sr, levels_map):
    processed = audio
    for eff in EFFECT_ORDER:
        lvl = int(levels_map.get(eff, 1))
        processed = apply_effect_by_name(processed, sr, eff, lvl)
    return processed

# ============== Feature Extraction Logic ==============


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


def extract_features_fx_scenarios(df, processor, model, model_type, dataset_name, target_sr, max_duration=15):
    """
    Build features for the scenario chains (Original + 3 bands).
    """
    records = []

    print("Extracting Scenario Features:")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scenarios"):
        audio, sr = librosa.load(row["path"], sr=None, mono=True)
        audio = audio[:sr * max_duration]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        for scen_name, levels_map in SCENARIOS.items():
            # No effects for "Original"
            if scen_name == "Original":
                processed = audio
            else:
                processed = apply_scenario_chain(audio, sr, levels_map)

            embed = get_embed(processed, processor, model, model_type, target_sr)
            rec = {
                "path": row["path"],
                "features": embed,
                "scenario": scen_name,
                "effect": "scenario",
                "level": -1
            }

            for eff in EFFECT_ORDER:
                rec[f"level_{eff}"] = int(levels_map.get(eff, 0))
            records.append(rec)

    df_fx_scen = pd.DataFrame(records)
    print("\nScenario features extracted")

    out_path = os.path.join(DATA_DIR, f"{dataset_name}_{model_type}_features_fx_scenarios.pkl")
    df_fx_scen.to_pickle(out_path)
    print(f"Saved scenario features to {out_path}")
    return df_fx_scen

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


def run_regression_pipeline_fx_scenarios(df, model_tag, reg_valence, reg_arousal, scaler):
    X_raw = np.vstack(df["features"].values)
    X = scaler.transform(X_raw)
    
    df["valence"] = reg_valence.predict(X)
    df["arousal"] = reg_arousal.predict(X)
    
    out_path = os.path.join(DATA_DIR, f"{model_tag}_results_fx_scenarios.pkl")
    df.to_pickle(out_path)
    print(f"Saved scenario regression results to {out_path}")

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


def run_classification_pipeline_fx_scenarios(df, model_tag, model, label_enc, scaler):
    X_raw = np.vstack(df["features"].values)
    X = scaler.transform(X_raw)
    
    preds = model.predict(X)
    df["results"] = label_enc.inverse_transform(preds)
    
    out_path = os.path.join(DATA_DIR, f"{model_tag}_results_fx_scenarios.pkl")
    df.to_pickle(out_path)
    print(f"Saved scenario classification results to {out_path}")

# ============== Multilabel Classification Logic ==============

def load_multilabel_pipeline_fx(model_tag, pickle_name=None, scaler_name=None, mlb_name=None):
    # scaler
    with open(os.path.join(MODEL_DIR, f"{model_tag}_scaler.pkl" if scaler_name is None else scaler_name), "rb") as f:
        scaler = pickle.load(f)

    # multilabel binarizer
    with open(os.path.join(MODEL_DIR, f"{model_tag}_mlb.pkl" if mlb_name is None else mlb_name), "rb") as f:
        mlb = pickle.load(f)

    # classifier (OneVsRestClassifier over XGBClassifier) saved as pickle
    with open(os.path.join(MODEL_DIR, f"{model_tag}_clf.pkl" if pickle_name is None else pickle_name), "rb") as f:
        clf = pickle.load(f)

    return scaler, mlb, clf


def run_multilabel_pipeline_fx(df, model_tag, clf, mlb, scaler, threshold=0.5):
    X_raw = np.vstack(df["features"].values)
    X = scaler.transform(X_raw)

    prob = clf.predict_proba(X)
    Y_proba = np.vstack(prob).T if isinstance(prob, list) else prob
    Y_pred = (Y_proba >= threshold).astype(int)

    # add columns for readability: proba_[label], pred_[label]
    for j, label in enumerate(mlb.classes_):
        df[f"proba_{label}"] = Y_proba[:, j]
        df[f"pred_{label}"] = Y_pred[:, j]

    # also store a semicolon-joined prediction string per row
    pred_labels_lists = mlb.inverse_transform(Y_pred)
    df["pred_labels"] = [";".join(lbls) if len(lbls) > 0 else "" for lbls in pred_labels_lists]

    print("Results Extracted (multilabel)")

    out_path = os.path.join(DATA_DIR, f"{model_tag}_results_fx.pkl")
    df.to_pickle(out_path)
    print(f"Saved results to {out_path}")


def run_multilabel_pipeline_fx_scenarios(df, model_tag, clf, mlb, scaler, threshold=0.5):
    X_raw = np.vstack(df["features"].values)
    X = scaler.transform(X_raw)
    
    prob = clf.predict_proba(X)
    Y_proba = np.vstack(prob).T if isinstance(prob, list) else prob
    Y_pred = (Y_proba >= threshold).astype(int)
    
    # add columns for readability: proba_[label], pred_[label]
    for j, label in enumerate(mlb.classes_):
        df[f"proba_{label}"] = Y_proba[:, j]
        df[f"pred_{label}"] = Y_pred[:, j]

    # also store a semicolon-joined prediction string per row
    pred_labels_lists = mlb.inverse_transform(Y_pred)
    df["pred_labels"] = [";".join(lbls) if len(lbls) > 0 else "" for lbls in pred_labels_lists]
    
    out_path = os.path.join(DATA_DIR, f"{model_tag}_results_fx_scenarios.pkl")
    df.to_pickle(out_path)
    print("Saved scenario multilabel results to", out_path)

# ============== Pipeline Definition ==============

def run_pipeline_fx(dataset_name, dataset_dir, model_type, processor, model, target_sr):
    print(f"\n=== Running: {model_type.upper()} + {dataset_name.upper()} ===")

    # 1) load and sample dataset
    df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))

    # 2) extract FX-augmented embeddings
    features_df = extract_features_fx(df, processor, model, model_type, dataset_name, target_sr)

    # 3) run the right downstream pipeline
    model_tag = f"{dataset_name}_{model_type}"
    if dataset_name in {"deam", "wtf_va"}:
        scaler, reg_v, reg_a = load_regression_pipeline_fx(model_tag)
        run_regression_pipeline_fx(features_df, model_tag, reg_v, reg_a, scaler)
    elif dataset_name == "emopia":
        scaler, label_enc, clf = load_classification_pipeline_fx(model_tag)
        run_classification_pipeline_fx(features_df, model_tag, clf, label_enc, scaler)
    else:  # wtf_lb
        scaler, mlb, clf = load_multilabel_pipeline_fx(model_tag)
        run_multilabel_pipeline_fx(features_df, model_tag, clf, mlb, scaler)


def run_pipeline_fx_scenarios(dataset_name, dataset_dir, model_type, processor, model, target_sr):
    print(f"\n=== Running SCENARIOS: {model_type.upper()} + {dataset_name.upper()} ===")
    
    # 1) load and sample dataset
    df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    
    # 2) extract FX-augmented embeddings
    scen_df = extract_features_fx_scenarios(df, processor, model, model_type, dataset_name, target_sr)
    
    # 3) run the right downstream pipeline
    model_tag = f"{dataset_name}_{model_type}"
    if dataset_name in {"deam", "wtf_va"}:
        scaler, reg_v, reg_a = load_regression_pipeline_fx(model_tag)
        run_regression_pipeline_fx_scenarios(scen_df, model_tag, reg_v, reg_a, scaler)
    elif dataset_name == "emopia":
        scaler, label_enc, clf = load_classification_pipeline_fx(model_tag)
        run_classification_pipeline_fx_scenarios(scen_df, model_tag, clf, label_enc, scaler)
    else:  # wtf_lb
        scaler, mlb, clf = load_multilabel_pipeline_fx(model_tag)
        run_multilabel_pipeline_fx_scenarios(scen_df, model_tag, clf, mlb, scaler)


def extract_results(mert, mert_proc, mert_sr, clap, clap_proc, clap_sr, qwen, qwen_proc, qwen_sr):
    run_pipeline_fx("deam", deam_dir, "mert", mert_proc, mert, mert_sr)
    run_pipeline_fx("deam", deam_dir, "clap", clap_proc, clap, clap_sr)
    run_pipeline_fx("deam", deam_dir, "qwen", qwen_proc, qwen, qwen_sr)
    run_pipeline_fx("emopia", emopia_dir, "mert", mert_proc, mert, mert_sr)
    run_pipeline_fx("emopia", emopia_dir, "clap", clap_proc, clap, clap_sr)
    run_pipeline_fx("emopia", emopia_dir, "qwen", qwen_proc, qwen, qwen_sr)
    run_pipeline_fx("wtf_va", wtf_dir, "mert", mert_proc, mert, mert_sr)
    run_pipeline_fx("wtf_va", wtf_dir, "clap", clap_proc, clap, clap_sr)
    run_pipeline_fx("wtf_va", wtf_dir, "qwen", qwen_proc, qwen, qwen_sr)
    run_pipeline_fx("wtf_lb", wtf_dir, "mert", mert_proc, mert, mert_sr)
    run_pipeline_fx("wtf_lb", wtf_dir, "clap", clap_proc, clap, clap_sr)
    run_pipeline_fx("wtf_lb", wtf_dir, "qwen", qwen_proc, qwen, qwen_sr)
    print("Results extraction complete!\n")
    
    # Scenarios
    run_pipeline_fx_scenarios("wtf_va", wtf_dir, "mert", mert_proc, mert, mert_sr)
    run_pipeline_fx_scenarios("wtf_va", wtf_dir, "clap", clap_proc, clap, clap_sr)
    run_pipeline_fx_scenarios("wtf_va", wtf_dir, "qwen", qwen_proc, qwen, qwen_sr)
    run_pipeline_fx_scenarios("wtf_lb", wtf_dir, "mert", mert_proc, mert, mert_sr)
    run_pipeline_fx_scenarios("wtf_lb", wtf_dir, "clap", clap_proc, clap, clap_sr)
    run_pipeline_fx_scenarios("wtf_lb", wtf_dir, "qwen", qwen_proc, qwen, qwen_sr)
    print("Scenario results extraction complete!\n")