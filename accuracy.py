from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
ACCURACY_DIR = os.path.join(OUTPUT_DIR, "accuracy")
NUM_LEVEL = int(os.getenv("NUM_LEVEL"))
DATASETS = ["deam", "emopia"]
MODEL_TYPES = ["mert", "clap", "qwen"]
EFFECTS = ["reverb", "delay", "distortion", "eq", "chorus", "phaser"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ACCURACY_DIR, exist_ok=True)


def compute_accuracy():
    all_metrics = {ds: {fx: {} for fx in EFFECTS} for ds in DATASETS}

    for dataset in DATASETS:
        for model in MODEL_TYPES:
            print(f"Computing metrics for {dataset} - {model}")

            # Load original labels
            original_df = pd.read_pickle(os.path.join(DATA_DIR, f"{dataset}_{model}_features.pkl")).drop(columns=["features"])
            if dataset == "deam":
                original_df.rename(columns={"valence": "original_valence", "arousal": "original_arousal"}, inplace=True)
            else:
                original_df.rename(columns={"emo_class": "original_class"}, inplace=True)

            # Load results
            results_df = pd.read_pickle(os.path.join(DATA_DIR, f"{dataset}_{model}_results_fx.pkl")).drop(columns=["features"])
            if dataset == "deam":
                results_df.rename(columns={"valence": "results_valence", "arousal": "results_arousal"}, inplace=True)
            else:
                results_df.rename(columns={"results": "results_class"}, inplace=True)

            # Merge data
            merged_df = pd.merge(original_df, results_df, on="path", how="inner")

            # Compute metrics for each effect and level
            for effect in EFFECTS:
                accs = []
                for level in range(1, NUM_LEVEL + 1):
                    df = merged_df[(merged_df["effect"] == effect) & (merged_df["level"] == level)]
                    if df.empty:
                        continue

                    if dataset == "deam":
                        accs.append((
                            level,
                            mean_squared_error(df["original_valence"], df["results_valence"]),
                            mean_absolute_error(df["original_valence"], df["results_valence"]),
                            r2_score(df["original_valence"], df["results_valence"]),
                            mean_squared_error(df["original_arousal"], df["results_arousal"]),
                            mean_absolute_error(df["original_arousal"], df["results_arousal"]),
                            r2_score(df["original_arousal"], df["results_arousal"])
                        ))
                    else:
                        accs.append((
                            level,
                            accuracy_score(df["original_class"], df["results_class"]),
                            precision_score(df["original_class"], df["results_class"], average="weighted", zero_division=0),
                            recall_score(df["original_class"], df["results_class"], average="weighted", zero_division=0),
                            f1_score(df["original_class"], df["results_class"], average="weighted", zero_division=0)
                        ))

                all_metrics[dataset][effect][model] = accs

    # Plot metrics
    for dataset in DATASETS:
        for effect in EFFECTS:
            metrics_info = [
                ("MSE Valence", 1), ("MAE Valence", 2), ("R2 Valence", 3),
                ("MSE Arousal", 4), ("MAE Arousal", 5), ("R2 Arousal", 6)
            ] if dataset == "deam" else [
                ("Accuracy", 1), ("Precision", 2), ("Recall", 3), ("F1 Score", 4)
            ]

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


compute_accuracy()

# from dotenv import load_dotenv
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score


# load_dotenv()
# DATA_DIR = os.getenv("DATA_DIR")
# OUTPUT_DIR = os.getenv("OUTPUT_DIR")
# ACCURACY_DIR = os.path.join(OUTPUT_DIR, "accuracy/")
# NUM_LEVEL = int(os.getenv("NUM_LEVEL"))
# DATASETS = ["deam", "emopia"]
# MODEL_TYPES = ["mert", "clap", "qwen"]
# EFFECTS = ["reverb", "delay", "distortion", "eq", "chorus", "phaser"]

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(ACCURACY_DIR, exist_ok=True)


# def compute_accuracy():
#     for dataset in DATASETS:
#         for model in MODEL_TYPES:
#             print(f"Computing accuracy for {dataset} - {model}")
            
#             # Original Df
#             original_df = pd.read_pickle(os.path.join(DATA_DIR, f"{dataset}_{model}_features.pkl"))
#             # Drop the feature column as it is not needed
#             original_df = original_df.drop(columns=["features"])
#             # Rename the columns that contain the original metrics
#             if dataset == "deam":
#                 original_df.rename(columns={"valence": "original_valence", "arousal": "original_arousal"}, inplace=True)
#             elif dataset == "emopia":
#                 original_df.rename(columns={"emo_class": "original_class"}, inplace=True)
            
#             # Results Df
#             results_df = pd.read_pickle(os.path.join(DATA_DIR, f"{dataset}_{model}_results_fx.pkl"))
#             # Drop the feature column as it is not needed
#             results_df = results_df.drop(columns=["features"])
#             # Rename the columns that contain the results metrics
#             if dataset == "deam":
#                 results_df.rename(columns={"valence": "results_valence", "arousal": "results_arousal"}, inplace=True)
#             elif dataset == "emopia":
#                 results_df.rename(columns={"results": "results_class"}, inplace=True)
            
#             # Merge the original and results dataframes
#             merged_df = pd.merge(original_df, results_df, on="path", how="inner")
            
#             for effect in EFFECTS:
#                 accuracies = []
#                 for level in range(1, NUM_LEVEL + 1):
#                     # Filter the merged dataframe for the current effect and level
#                     filtered_df = merged_df[(merged_df["effect"] == effect) & (merged_df["level"] == level)]
#                     if filtered_df.empty:
#                         print(f"No data for {effect} at level {level} in {dataset} - {model}")
#                         continue
#                     # Calculate accuracy
#                     if dataset == "deam":
#                         # DEAM is a regression task, so we compute the mse, mae, and r2 score
#                         mse_valence = mean_squared_error(filtered_df["original_valence"], filtered_df["results_valence"])
#                         mse_arousal = mean_squared_error(filtered_df["original_arousal"], filtered_df["results_arousal"])
#                         mae_valence = mean_absolute_error(filtered_df["original_valence"], filtered_df["results_valence"])
#                         mae_arousal = mean_absolute_error(filtered_df["original_arousal"], filtered_df["results_arousal"])
#                         r2_valence = r2_score(filtered_df["original_valence"], filtered_df["results_valence"])
#                         r2_arousal = r2_score(filtered_df["original_arousal"], filtered_df["results_arousal"])
#                         accuracies.append((effect, level, mse_valence, mse_arousal, mae_valence, mae_arousal, r2_valence, r2_arousal))
#                     elif dataset == "emopia":
#                         # Emopia is a classification task, so we compute the accuracy, precision, recall, and f1 score
#                         accuracy = accuracy_score(filtered_df["original_class"], filtered_df["results_class"])
#                         precision = precision_score(filtered_df["original_class"], filtered_df["results_class"], average='weighted', zero_division=0)
#                         # recall = recall_score(filtered_df["original_class"], filtered_df["results_class"], average='weighted', zero_division=0).  # The same as accuracy
#                         f1 = f1_score(filtered_df["original_class"], filtered_df["results_class"], average='weighted', zero_division=0)
#                         accuracies.append((effect, level, accuracy, precision, f1))

#                 # Plot the accuracies in a diagram
#                 if accuracies:
#                     plt.figure(figsize=(10, 6))
#                     if dataset == "deam":
#                         mse_vals = [acc[2] for acc in accuracies]
#                         mae_vals = [acc[4] for acc in accuracies]
#                         r2_vals = [acc[6] for acc in accuracies]
#                         plt.plot([acc[1] for acc in accuracies], mse_vals, label='MSE Valence', marker='o')
#                         plt.plot([acc[1] for acc in accuracies], mae_vals, label='MAE Valence', marker='o')
#                         plt.plot([acc[1] for acc in accuracies], r2_vals, label='R2 Valence', marker='o')
#                         plt.title(f"{dataset} - {model} - {effect} Accuracy")
#                         plt.xlabel("Level")
#                         plt.ylabel("Score")
#                         plt.legend()
#                         plt.savefig(os.path.join(ACCURACY_DIR, f"{dataset}_{model}_{effect}_accuracy.png"))
#                     elif dataset == "emopia":
#                         accuracy_vals = [acc[2] for acc in accuracies]
#                         precision_vals = [acc[3] for acc in accuracies]
#                         f1_vals = [acc[4] for acc in accuracies]
#                         plt.plot([acc[1] for acc in accuracies], accuracy_vals, label='Accuracy', marker='o')
#                         plt.plot([acc[1] for acc in accuracies], precision_vals, label='Precision', marker='o')
#                         plt.plot([acc[1] for acc in accuracies], f1_vals, label='F1 Score', marker='o')
#                         plt.title(f"{dataset} - {model} - {effect} Accuracy")
#                         plt.xlabel("Level")
#                         plt.ylabel("Score")
#                         plt.legend()
#                         plt.savefig(os.path.join(ACCURACY_DIR, f"{dataset}_{model}_{effect}_accuracy.png"))
#                     plt.close()