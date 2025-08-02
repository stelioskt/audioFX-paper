from dotenv import load_dotenv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns


load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
NUM_LEVEL = os.getenv("NUM_LEVEL")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================== Stats ======================

def compute_stats_deam(deam, model):
    # Group by effect name and level
    deam_summary = deam.groupby(['effect', 'level']).agg(
        mean_valence=('valence', 'mean'),
        std_valence=('valence', 'std'),
        mean_arousal=('arousal', 'mean'),
        std_arousal=('arousal', 'std')
    ).reset_index()

    # Print results
    deam_summary.to_csv(os.path.join(OUTPUT_DIR, f"{model}_deam_summary.csv"), index=False)

    return deam_summary


def compute_stats_emopia(emopia, model):
    # Group by effect name and level to calculate label frequencies
    emopia_summary = emopia.groupby(['effect', 'level', 'results']).size().reset_index(name='count')

    # Pivot for stacked bar chart
    emopia_pivot = emopia_summary.pivot(index=['effect', 'level'], columns='results', values='count').fillna(0)

    # Normalize for proportions
    emopia_pivot_normalized = emopia_pivot.div(emopia_pivot.sum(axis=1), axis=0)

    # Print results
    emopia_pivot_normalized.to_csv(os.path.join(OUTPUT_DIR, f"{model}_emopia_summary.csv"))

    return emopia_pivot_normalized


# ====================== Regression Plots ======================

def plot_heatmap_deam(deam_summary, prefix):
    plt.figure(figsize=(10, 6))
    heatmap_data = deam_summary.pivot_table(index='effect', columns='level', values='mean_valence')
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mean Valence by Effect and Level')
    plt.xlabel('Level')
    plt.ylabel('Effect Name')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_mean_valence_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    heatmap_data = deam_summary.pivot_table(index='effect', columns='level', values='mean_arousal')
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mean Arousal by Effect and Level')
    plt.xlabel('Level')
    plt.ylabel('Effect Name')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_mean_arousal_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_va_trends(deam_summary, prefix):
    plt.figure(figsize=(12, 6))
    for effect in deam_summary['effect'].unique():
        effect_data = deam_summary[deam_summary['effect'] == effect]
        plt.plot(effect_data['level'], effect_data['mean_valence'], label=f'{effect} (Valence)')
        plt.plot(effect_data['level'], effect_data['mean_arousal'], linestyle='--', label=f'{effect} (Arousal)')

    plt.title('Valence and Arousal Trends by Effect and Level')
    plt.xlabel('Level')
    plt.ylabel('Mean Values')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_va_trends.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_va_plane(deam, prefix):
    colormap = cm.get_cmap('viridis', 5)  # 5 level levels (0 to 4)

    for effect in deam['effect'].unique():
        plt.figure(figsize=(8, 6))
        effect_data = deam[deam['effect'] == effect]

        for level in sorted(effect_data['level'].unique()):
            level_data = effect_data[effect_data['level'] == level]
            color = colormap(level / (NUM_LEVEL+1))
            plt.scatter(
                level_data['valence'],
                level_data['arousal'],
                label=f'Level {level}',
                color=color,
                alpha=0.7
            )

        plt.title(f'Valence-Arousal Plane for {effect}', fontsize=14)
        plt.xlabel('Valence', fontsize=12)
        plt.ylabel('Arousal', fontsize=12)
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.7)
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.7)
        plt.grid(alpha=0.3)
        plt.legend(title='Level', fontsize=10)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_va_plane_{effect}.png"), dpi=300, bbox_inches='tight')
        plt.close()


# ====================== Classification Plots ======================

def plot_label_proportions(emopia_pivot_normalized, prefix):
    effect_list = emopia_pivot_normalized.index.get_level_values('effect').unique()

    for effect in effect_list:
        data = emopia_pivot_normalized.loc[effect]
        data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.title(f'Label Proportions for {effect}')
        plt.xlabel('Level')
        plt.ylabel('Proportion')
        plt.legend(title='Label')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_label_proportions_{effect}.png"), dpi=300, bbox_inches='tight')
        plt.close()


def radar_chart(data, title, filename):
    labels = data.columns
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for idx, row in data.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Level {idx}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_title(title, size=16, pad=30)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar_charts(emopia_pivot_normalized, prefix):
    for effect in emopia_pivot_normalized.index.get_level_values('effect').unique():
        radar_chart(emopia_pivot_normalized.loc[effect],
                    f'Radar Chart for {effect}',
                    os.path.join(OUTPUT_DIR, f"{prefix}_radar_{effect}.png"))


# ====================== Correlation & Chi-Square ======================

def correlation_matrix(deam, prefix):
    correlation = deam[['level', 'valence', 'arousal']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap: Level, Valence, and Arousal')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()


def chi_square(emopia):
    # Chi-square test for association between effect level and labels
    results = []
    for effect in emopia['effect'].unique():
        contingency_table = pd.crosstab(emopia[emopia['effect'] == effect]['level'],
                                        emopia[emopia['effect'] == effect]['results'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        results.append(f'{effect}: Chi-square = {chi2:.2f}, p-value = {p:.4f}')
    
    with open(os.path.join(OUTPUT_DIR, f"{prefix}_chi_square_results.txt"), "w") as f:
        f.write("\n".join(results))


def visualize_results():
    # Load the datasets
    deam_mert = pd.read_pickle(os.path.join(DATA_DIR, "deam_mert_results_fx.pkl"))
    deam_clap = pd.read_pickle(os.path.join(DATA_DIR, "deam_clap_results_fx.pkl"))
    deam_qwen = pd.read_pickle(os.path.join(DATA_DIR, "deam_qwen_results_fx.pkl"))
    emopia_mert = pd.read_pickle(os.path.join(DATA_DIR, "emopia_mert_results_fx.pkl"))
    emopia_clap = pd.read_pickle(os.path.join(DATA_DIR, "emopia_clap_results_fx.pkl"))
    emopia_qwen = pd.read_pickle(os.path.join(DATA_DIR, "emopia_qwen_results_fx.pkl"))

    # Compute statistics
    deam_mert_sum = compute_stats_deam(deam_mert, "mert")
    deam_clap_sum = compute_stats_deam(deam_clap, "clap")
    deam_qwen_sum = compute_stats_deam(deam_qwen, "qwen")

    emopia_mert_norm = compute_stats_emopia(emopia_mert, "mert")
    emopia_clap_norm = compute_stats_emopia(emopia_clap, "clap")
    emopia_qwen_norm = compute_stats_emopia(emopia_qwen, "qwen")

    # Regression Plots
    plot_heatmap_deam(deam_mert_sum, "mert")
    plot_heatmap_deam(deam_clap_sum, "clap")
    plot_heatmap_deam(deam_qwen_sum, "qwen")

    plot_va_trends(deam_mert_sum, "mert")
    plot_va_trends(deam_clap_sum, "clap")
    plot_va_trends(deam_qwen_sum, "qwen")

    plot_va_plane(deam_mert, "mert")
    plot_va_plane(deam_clap, "clap")
    plot_va_plane(deam_qwen, "qwen")

    # Classification Plots
    plot_label_proportions(emopia_mert_norm, "mert")
    plot_label_proportions(emopia_clap_norm, "clap")
    plot_label_proportions(emopia_qwen_norm, "qwen")

    plot_radar_charts(emopia_mert_norm, "mert")
    plot_radar_charts(emopia_clap_norm, "clap")
    plot_radar_charts(emopia_qwen_norm, "qwen")

    # Correlation Analysis
    correlation_matrix(deam_mert, "mert")
    correlation_matrix(deam_clap, "clap")
    correlation_matrix(deam_qwen, "qwen")

    chi_square(emopia_mert, "mert")
    chi_square(emopia_clap, "clap")
    chi_square(emopia_qwen, "qwen")
    print("Visualisation complete!\n")