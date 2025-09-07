from dotenv import load_dotenv
import librosa
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
import torch
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
import logging
import psutil
import time

load_dotenv()
emopia_dir = os.getenv("EMOPIA_DIR")
deam_dir = os.getenv("DEAM_DIR")
wtf_dir = os.getenv("WITHEFLOW_DIR")
DATA_DIR = os.getenv("DATA_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")
RND_STATE = int(os.getenv("RANDOM_STATE"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["joyfulActivation", "neutral", "nostalgia", "peacefulness", "power",
           "sadness", "tenderness", "tension", "transcendence", "wonder"]


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("outputs/pipeline.log"),
    ]
)

def log_resources(tag=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"[{tag}] CPU: {psutil.cpu_percent()}% | RAM: {mem_info.rss / 1024**2:.2f} MB")

def get_embed(audio, processor, model, model_type, target_sr):
    dtype = torch.float16 if model.dtype == torch.float16 else torch.float32
    tensor_audio = torch.tensor(audio, dtype=dtype)

    if model_type == "mert":
        inputs = processor(tensor_audio, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            hidden = out.last_hidden_state.squeeze(0)
            embed = hidden.mean(dim=0).cpu().numpy()

    elif model_type == "clap":
        inputs = processor(audios=audio, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            audio_embeds = model.get_audio_features(**inputs)
            embed = audio_embeds.squeeze(0).cpu().numpy()

    elif model_type == "qwen":
        inputs = processor(audios=audio, sampling_rate=target_sr, return_tensors="pt", text=processor.audio_token)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
            last_decoder_hidden = out.hidden_states[-1]
            embed = last_decoder_hidden.mean(dim=1).cpu().numpy()

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return embed


def extract_features(df, processor, model, model_type, target_sr, max_duration=15):
    features = []
    
    logging.info(f"Starting feature extraction for {model_type}...")
    start_time = time.time()

    print("Extracting Features:")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Features"):
        # 1. Load & preprocess audio
        audio, sr = librosa.load(row["path"], sr=None, mono=True)
        audio = audio[:sr * max_duration]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        embed = get_embed(audio, processor, model, model_type, target_sr)
        features.append(embed)
        
        if idx % 50 == 0:
            log_resources(tag=f"Feature {idx}/{len(df)}")

    df["features"] = features
    print(f"\nFeatures extracted")
    logging.info(f"Feature extraction complete in {time.time()-start_time:.2f} seconds")
    return df


def save_features(df, dataset_name, model_tag):
    out_path = os.path.join(DATA_DIR, f"{dataset_name}_{model_tag}_features.pkl")
    meta_cols = {"deam": ["valence", "arousal"],
                 "wtf_va": ["valence", "arousal"],
                 "emopia": ["emo_class"],
                 "wtf_lb": ["labels"]
                }[dataset_name]
    save_df = df[["path", "features"] + meta_cols].copy()
    save_df.to_pickle(out_path)
    print(f"Saved features to {out_path}")
    return out_path


# ================== Regression Logic ==================

def prepare_data_regression(df):
    X = np.vstack(df["features"].values)
    y = df[["valence", "arousal"]].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, scaler


def tune_xgb_reg(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1]
    }

    grid = GridSearchCV(XGBRegressor(objective="reg:squarederror",random_state=42, n_jobs=-1),
                        param_grid,
                        cv=3,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1,
                        verbose=1)

    grid.fit(X_train, y_train)

    return grid.best_estimator_


def run_regression_pipeline(df, dataset_name, model_type):
    model_tag = f"{dataset_name}_{model_type}"
    X_train, X_test, y_train, y_test, scaler = prepare_data_regression(df)
    result = []
    # Split targets
    for target_idx, name in enumerate(["Valence", "Arousal"]):
        y_tr = y_train[:, target_idx]
        y_te = y_test[:, target_idx]

        model = tune_xgb_reg(X_train, y_tr)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_te, preds)
        mse = mean_squared_error(y_te, preds)
        r2 = r2_score(y_te, preds)

        logging.info(f"[{model_tag}][{name}] MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
        model.save_model(os.path.join(MODEL_DIR, f"{model_tag}_{name.lower()}.json"))
        
        result.append({
            "model": model_type,
            "target": name,
            "mae": mae,
            "mse": mse,
            "r2": r2
        })

    with open(os.path.join(MODEL_DIR, f"{model_tag}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    return result


# ================== Classification Logic ==================

def prepare_data_classification(df):
    X = np.vstack(df["features"].values)
    y = df["emo_class"].values

    label_enc = LabelEncoder().fit(y)
    y_enc = label_enc.transform(y)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_enc, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_enc, scaler


def tune_xgb_clf(X_train, y_train, X_val, y_val):
    param_grid = {"n_estimators": [100, 200],
                  "max_depth": [3, 5],
                  "learning_rate": [0.01, 0.1]}

    clf = XGBClassifier(objective="multi:softmax",
                        num_class=4,
                        random_state=42,
                        eval_metric="mlogloss",
                        n_jobs=-1)

    grid = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_


def run_classification_pipeline(df, dataset_name, model_type):
    model_tag = f"{dataset_name}_{model_type}"
    X_train, X_val, X_test, y_train, y_val, y_test, label_enc, scaler = prepare_data_classification(df)
    model = tune_xgb_clf(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    logging.info(f"[{model_tag}][Classification] Accuracy: {acc:.4f}")
    logging.info(classification_report(label_enc.inverse_transform(y_test),
                                label_enc.inverse_transform(preds)))
    model.save_model(os.path.join(MODEL_DIR, f"{model_tag}_clf.json"))
    with open(os.path.join(MODEL_DIR, f"{model_tag}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, f"{model_tag}_enc.pkl"), "wb") as f:
        pickle.dump(label_enc, f)
    
    return {
        "model": model_type,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


# ============= Multi-label Classification Logic =============

def prepare_data_multilabel(df, labels_col="labels", classes=None):
    """
    df[labels_col] should contain iterables (list/tuple/set) of label strings, or stringified lists.
    """
    y_list = df[labels_col].apply(lambda s: s.split(";"))
    mlb = MultiLabelBinarizer(classes=classes)
    Y = mlb.fit_transform(y_list)

    X = np.vstack(df["features"].values)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_scaled, Y, test_size=0.3, random_state=42
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, mlb, scaler


def run_multilabel_pipeline(df, dataset_name, model_type, labels_col="labels", classes=None, threshold=0.5):
    model_tag = f"{dataset_name}_{model_type}"
    X_train, X_val, X_test, Y_train, Y_val, Y_test, mlb, scaler = prepare_data_multilabel(
        df, labels_col=labels_col, classes=classes
    )

    base_xgb = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=RND_STATE,
        eval_metric="logloss",
        tree_method="hist"
    )
    clf = OneVsRestClassifier(base_xgb)
    clf.fit(X_train, Y_train)

    Y_test_pred = (clf.predict_proba(X_test) >= threshold).astype(int)
    f1_micro = f1_score(Y_test, Y_test_pred, average="micro", zero_division=0)
    f1_macro = f1_score(Y_test, Y_test_pred, average="macro", zero_division=0)
    logging.info(f"[{model_tag}][Multilabel] F1-micro: {f1_micro:.4f} | F1-macro: {f1_macro:.4f}")
    
    per_label_acc_array = (Y_test == Y_test_pred).mean(axis=0)
    per_label_acc = dict(zip(mlb.classes_, per_label_acc_array))

    with open(os.path.join(MODEL_DIR, f"{model_tag}_clf.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, f"{model_tag}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, f"{model_tag}_mlb.pkl"), "wb") as f:
        pickle.dump(mlb, f)

    return {
        "model": model_type,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "n_labels": len(mlb.classes_),
        "per_label_accuracy": per_label_acc
    }

# ================== Pipeline Definition ==================

def run_pipeline(dataset_name, dataset_dir, model_type, processor, model, target_sr):
    print(f"\n=== Running: {model_type.upper()} + {dataset_name.upper()} ===")
    df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    df = extract_features(df, processor, model, model_type, target_sr)
    save_features(df, dataset_name, model_type)

    if dataset_name in {"deam", "wtf_va"}:
        return run_regression_pipeline(df, dataset_name, model_type)
    elif dataset_name == "emopia":
        return run_classification_pipeline(df, dataset_name, model_type)
    else:   # wtf_lb
        return run_multilabel_pipeline(df, dataset_name, model_type, classes=classes)


def model_eval(mert, mert_proc, mert_sr, clap, clap_proc, clap_sr, qwen, qwen_proc, qwen_sr):
    deam_results, emopia_results, wtf_va_results, wtf_lb_results = [], [], [], []

    deam_results.append(run_pipeline("deam", deam_dir, "mert", mert_proc, mert, mert_sr))
    deam_results.append(run_pipeline("deam", deam_dir, "clap", clap_proc, clap, clap_sr))
    deam_results.append(run_pipeline("deam", deam_dir, "qwen", qwen_proc, qwen, qwen_sr))
    emopia_results.append(run_pipeline("emopia", emopia_dir, "mert", mert_proc, mert, mert_sr))
    emopia_results.append(run_pipeline("emopia", emopia_dir, "clap", clap_proc, clap, clap_sr))
    emopia_results.append(run_pipeline("emopia", emopia_dir, "qwen", qwen_proc, qwen, qwen_sr))
    wtf_va_results.append(run_pipeline("wtf_va", wtf_dir, "mert", mert_proc, mert, mert_sr))
    wtf_va_results.append(run_pipeline("wtf_va", wtf_dir, "clap", clap_proc, clap, clap_sr))
    wtf_va_results.append(run_pipeline("wtf_va", wtf_dir, "qwen", qwen_proc, qwen, qwen_sr))
    wtf_lb_results.append(run_pipeline("wtf_lb", wtf_dir, "mert", mert_proc, mert, mert_sr))
    wtf_lb_results.append(run_pipeline("wtf_lb", wtf_dir, "clap", clap_proc, clap, clap_sr))
    wtf_lb_results.append(run_pipeline("wtf_lb", wtf_dir, "qwen", qwen_proc, qwen, qwen_sr))
    
    # Save the results
    with open(os.path.join(DATA_DIR, "deam_results.pkl"), "wb") as f:
        pickle.dump(deam_results, f)
    with open(os.path.join(DATA_DIR, "emopia_results.pkl"), "wb") as f:
        pickle.dump(emopia_results, f)
    with open(os.path.join(DATA_DIR, "wtf_va_results.pkl"), "wb") as f:
        pickle.dump(wtf_va_results, f)
    with open(os.path.join(DATA_DIR, "wtf_lb_results.pkl"), "wb") as f:
        pickle.dump(wtf_lb_results, f)

    print("Model evaluation complete!\n")