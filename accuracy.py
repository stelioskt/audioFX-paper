from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from typing import List


load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
ACCURACY_DIR = os.path.join(OUTPUT_DIR, "accuracy")

NUM_LEVEL = int(os.getenv("NUM_LEVEL"))
DATASETS = ["deam", "emopia", "wtf_va", "wtf_lb"]
MODEL_TYPES = ["mert", "clap", "qwen"]
EFFECTS = ["reverb", "delay", "distortion", "eq", "chorus", "phaser"]
SCENARIO_ORDER = ["Original", "Pink Floyd", "Rage Against The Machine", "U2"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ACCURACY_DIR, exist_ok=True)

WTF_LB_CLASSES = [
    "sadness", "nostalgia", "peacefulness", "neutral",
    "tenderness", "joyfulActivation", "wonder", "transcendence",
    "power", "tension"
]

def str_to_multihot(label_str: str, classes: List[str]) -> List[int]:
    """Turn 'a;b;c' into multihot [0/1,...] in the given class order."""
    labs = set(x.strip() for x in str(label_str).split(";") if x and x.strip())
    return [1 if cls in labs else 0 for cls in classes]

def series_to_multihot(series: pd.Series, classes: List[str]) -> pd.DataFrame:
    """Apply str_to_multihot to a Series of semicolon strings -> DataFrame of 0/1."""
    arr = series.apply(lambda s: str_to_multihot(s, classes)).tolist()
    return pd.DataFrame(arr, columns=classes, index=series.index)



def accuracy():
    all_metrics = {ds: {fx: {} for fx in EFFECTS} for ds in DATASETS}

    for dataset in DATASETS:
        for model in MODEL_TYPES:
            print(f"Computing metrics for {dataset} - {model}")

            # Load original labels
            original_df = pd.read_pickle(os.path.join(DATA_DIR, f"{dataset}_{model}_features.pkl")).drop(columns=["features"])
            if dataset in {"deam", "wtf_va"}:
                original_df.rename(columns={"valence": "original_valence", "arousal": "original_arousal"}, inplace=True)
            elif dataset == "emopia":
                original_df.rename(columns={"emo_class": "original_class"}, inplace=True)
            else:  # wtf_lb
                original_df.rename(columns={"labels": "original_labels"}, inplace=True)

            # Load results
            results_df = pd.read_pickle(os.path.join(DATA_DIR, f"{dataset}_{model}_results_fx.pkl")).drop(columns=["features"])
            if dataset in {"deam", "wtf_va"}:
                results_df.rename(columns={"valence": "results_valence", "arousal": "results_arousal"}, inplace=True)
            elif dataset == "emopia":
                results_df.rename(columns={"results": "results_class"}, inplace=True)
            else:  # wtf_lb
                pass

            # Merge data
            merged_df = pd.merge(original_df, results_df, on="path", how="inner")

            # Compute metrics for each effect and level
            for effect in EFFECTS:
                accs = []
                for level in range(1, NUM_LEVEL + 1):
                    df = merged_df[(merged_df["effect"] == effect) & (merged_df["level"] == level)]
                    if df.empty:
                        continue

                    if dataset in {"deam", "wtf_va"}:
                        accs.append((
                            level,
                            mean_squared_error(df["original_valence"], df["results_valence"]),
                            mean_absolute_error(df["original_valence"], df["results_valence"]),
                            r2_score(df["original_valence"], df["results_valence"]),
                            mean_squared_error(df["original_arousal"], df["results_arousal"]),
                            mean_absolute_error(df["original_arousal"], df["results_arousal"]),
                            r2_score(df["original_arousal"], df["results_arousal"])
                        ))
                    elif dataset == "emopia":
                        accs.append((
                            level,
                            accuracy_score(df["original_class"], df["results_class"]),
                            precision_score(df["original_class"], df["results_class"], average="weighted", zero_division=0),
                            recall_score(df["original_class"], df["results_class"], average="weighted", zero_division=0),
                            f1_score(df["original_class"], df["results_class"], average="weighted", zero_division=0)
                        ))
                    else:  # wtf_lb
                        Y_true = series_to_multihot(df["original_labels"], WTF_LB_CLASSES).to_numpy()
                        if "pred_labels" in df.columns:
                            Y_pred = series_to_multihot(df["pred_labels"], WTF_LB_CLASSES).to_numpy()
                        else:
                            # Fallback: derive from per-label columns if pred_labels is absent
                            # Look for columns named pred_<class> in any case variant
                            bin_cols = []
                            for cls in WTF_LB_CLASSES:
                                # find matching pred_<cls> column case-insensitively
                                cand = [c for c in df.columns if c.lower() == f"pred_{cls.lower()}"]
                                if cand:
                                    bin_cols.append(cand[0])
                                else:
                                    raise KeyError(f"Missing predicted column for class '{cls}'")
                            Y_pred = df[bin_cols].astype(int).to_numpy()

                        f1_micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
                        f1_macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
                        
                        # Keep tuple length small to minimize plotting changes
                        accs.append((level, f1_micro, f1_macro))

                all_metrics[dataset][effect][model] = accs

    # Plot metrics
    for dataset in DATASETS:
        if dataset in {"deam", "wtf_va"}:
            metrics_info = [
                ("MSE Valence", 1), ("MAE Valence", 2), ("R2 Valence", 3),
                ("MSE Arousal", 4), ("MAE Arousal", 5), ("R2 Arousal", 6)
            ]
        elif dataset == "emopia":
            metrics_info = [
                ("Accuracy", 1), ("Precision", 2), ("Recall", 3), ("F1 Score", 4)
            ]
        else:
            metrics_info = [
                ("F1 Micro", 1), ("F1 Macro", 2)
            ]
            
        for effect in EFFECTS:

            for metric_name, idx in metrics_info:
                plt.figure(figsize=(8, 5))
                plotted = False
                for model in MODEL_TYPES:
                    accs = all_metrics[dataset][effect].get(model, [])
                    if not accs:
                        continue
                    plt.plot([a[0] for a in accs], [a[idx] for a in accs], marker="o", label=model)
                    plotted = True

                if plotted:
                    plt.title(f"{dataset.upper()} – {effect.capitalize()} – {metric_name}")
                    plt.xlabel("Level")
                    plt.ylabel(metric_name)
                    plt.legend(title="Model")
                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.savefig(os.path.join(ACCURACY_DIR, f"{dataset}_{effect}_{metric_name.replace(' ','_')}.png"), bbox_inches="tight")
                    plt.close()
                    
    print("Accuracy computation and plotting complete.\n")


def accuracy_scenarios():
    for dataset in ["wtf_va", "wtf_lb"]:
        print(f"[SCENARIOS] Computing metrics for {dataset}")

        # collect metrics per model
        results_per_model = {}

        for model in MODEL_TYPES:
            # original labels
            orig = pd.read_pickle(os.path.join(DATA_DIR, f"{dataset}_{model}_features.pkl")).drop(columns=["features"])
            if dataset == "wtf_va":
                orig = orig.rename(columns={"valence": "original_valence", "arousal": "original_arousal"})
            else:
                orig = orig.rename(columns={"labels": "original_labels"})

            # scenario predictions
            scen = pd.read_pickle(os.path.join(DATA_DIR, f"{dataset}_{model}_results_fx_scenarios.pkl")).copy()

            merged = pd.merge(orig, scen, on="path", how="inner")

            rows = {}
            for scen_name, df_s in merged.groupby("scenario"):
                if dataset == "wtf_va":
                    mse_v = mean_squared_error(df_s["original_valence"], df_s["valence"])
                    mae_v = mean_absolute_error(df_s["original_valence"], df_s["valence"])
                    r2_v  = r2_score(df_s["original_valence"], df_s["valence"])

                    mse_a = mean_squared_error(df_s["original_arousal"], df_s["arousal"])
                    mae_a = mean_absolute_error(df_s["original_arousal"], df_s["arousal"])
                    r2_a  = r2_score(df_s["original_arousal"], df_s["arousal"])

                    rows[scen_name] = (mse_v, mae_v, r2_v, mse_a, mae_a, r2_a)
                else:
                    Y_true = series_to_multihot(df_s["original_labels"], WTF_LB_CLASSES).to_numpy()
                    if "pred_labels" in df_s.columns:
                        Y_pred = series_to_multihot(df_s["pred_labels"], WTF_LB_CLASSES).to_numpy()
                    else:
                        bin_cols = []
                        for cls in WTF_LB_CLASSES:
                            cand = [c for c in df_s.columns if c.lower() == f"pred_{cls.lower()}"]
                            if cand:
                                bin_cols.append(cand[0])
                        Y_pred = df_s[bin_cols].astype(int).to_numpy()

                    f1_micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
                    f1_macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
                    rows[scen_name] = (f1_micro, f1_macro)

            results_per_model[model] = rows

        # ---- Plot all models together ----
        if dataset == "wtf_va":
            metrics = [
                ("MSE Valence", 0), ("MAE Valence", 1), ("R2 Valence", 2),
                ("MSE Arousal", 3), ("MAE Arousal", 4), ("R2 Arousal", 5)
            ]
        else:
            metrics = [("F1 Micro", 0), ("F1 Macro", 1)]

        xs = SCENARIO_ORDER

        for metric_name, idx in metrics:
            plt.figure(figsize=(8, 5))
            for model in MODEL_TYPES:
                rows = results_per_model.get(model, {})
                vals = [rows[s][idx] if s in rows else np.nan for s in xs]
                plt.plot(xs, vals, marker="o", label=model)

            plt.title(f"{dataset.upper()} – SCENARIOS – {metric_name}")
            plt.xlabel("Scenario")
            plt.ylabel(metric_name)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(title="Model")
            plt.tight_layout()
            fname = f"{dataset}_SCENARIOS_{metric_name.replace(' ', '_')}.png"
            plt.savefig(os.path.join(ACCURACY_DIR, fname), bbox_inches="tight")
            plt.close()

    print("Scenario accuracy computation and plotting complete.\n")