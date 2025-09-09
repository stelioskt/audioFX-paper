from dotenv import load_dotenv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
from matplotlib import pyplot


load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
NUM_LEVEL = int(os.getenv("NUM_LEVEL"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
EMOTION_DIR = os.path.join(OUTPUT_DIR, "emotion/")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMOTION_DIR, exist_ok=True)

WTF_LB_CLASSES = [
    "sadness", "nostalgia", "peacefulness", "neutral",
    "tenderness", "joyfulActivation", "wonder", "transcendence",
    "power", "tension"
]

SCENARIO_ORDER = ["Original", "Pink Floyd", "Rage Against The Machine", "U2"]
SCENARIO_PALETTE = sns.color_palette("tab10", n_colors=len(SCENARIO_ORDER))
SCENARIO_COLORS = dict(zip(SCENARIO_ORDER, SCENARIO_PALETTE))

# ====================== Stats ======================

def compute_stats_reg(df, model, dataset):
    # Group by effect name and level
    summary = df.groupby(['effect', 'level']).agg(
        mean_valence=('valence', 'mean'),
        std_valence=('valence', 'std'),
        mean_arousal=('arousal', 'mean'),
        std_arousal=('arousal', 'std')
    ).reset_index()

    # Print results
    summary.to_csv(os.path.join(EMOTION_DIR, f"{model}_{dataset}_summary.csv"), index=False)

    return summary


def compute_stats_clf(df, model, dataset):
    # Group by effect name and level to calculate label frequencies
    summary = df.groupby(['effect', 'level', 'results']).size().reset_index(name='count')

    # Pivot for stacked bar chart
    pivot = summary.pivot(index=['effect', 'level'], columns='results', values='count').fillna(0)

    # Normalize for proportions
    pivot_normalized = pivot.div(pivot.sum(axis=1), axis=0)

    # Print results
    pivot_normalized.to_csv(os.path.join(EMOTION_DIR, f"{model}_{dataset}_summary.csv"))

    return pivot_normalized


def compute_stats_multilabel(df, model, dataset, classes=WTF_LB_CLASSES):
    df = df.copy()

    # Map classes to their corresponding pred_* columns
    pred_cols = {lab: f"pred_{lab.lower()}" for lab in classes}

    for lab in classes:
        col_candidates = [c for c in df.columns if c.lower() == f"pred_{lab.lower()}"]
        if col_candidates:
            pred_cols[lab] = col_candidates[0]
        else:
            raise KeyError(f"No prediction column found for class '{lab}'")

    # Group by effect/level and average predictions
    summary = df.groupby(["effect", "level"])[list(pred_cols.values())].mean().reset_index()

    # Rename columns back to class names
    summary = summary.rename(columns={v: k for k, v in pred_cols.items()})

    summary.to_csv(os.path.join(EMOTION_DIR, f"{model}_{dataset}_summary.csv"), index=False)

    pivot = summary.set_index(["effect", "level"])[classes]
    return pivot


def compute_stats_reg_scenarios(df, model, dataset):
    summary = df.groupby(['effect', 'scenario']).agg(
        mean_valence=('valence', 'mean'),
        std_valence=('valence', 'std'),
        mean_arousal=('arousal', 'mean'),
        std_arousal=('arousal', 'std')
    ).reset_index()

    # ensure scenario ordering for downstream plots
    cats = pd.Categorical(summary['scenario'], categories=SCENARIO_ORDER, ordered=True)
    summary['scenario'] = cats
    summary = summary.sort_values(['effect', 'scenario'])

    summary.to_csv(os.path.join(EMOTION_DIR, f"{model}_{dataset}_summary_scenarios.csv"), index=False)
    return summary


def compute_stats_multilabel_scenarios(df, model, dataset, classes=WTF_LB_CLASSES):
    df = df.copy()
    pred_cols = {lab: f"pred_{lab.lower()}" for lab in classes}
    for lab in classes:
        col_candidates = [c for c in df.columns if c.lower() == f"pred_{lab.lower()}"]
        if col_candidates:
            pred_cols[lab] = col_candidates[0]
        else:
            raise KeyError(f"No prediction column found for class '{lab}'")

    summary = df.groupby(["effect", "scenario"])[list(pred_cols.values())].mean().reset_index()
    summary = summary.rename(columns={v: k for k, v in pred_cols.items()})

    # enforce scenario order for consistency
    cats = pd.Categorical(summary['scenario'], categories=SCENARIO_ORDER, ordered=True)
    summary['scenario'] = cats
    summary = summary.sort_values(['effect', 'scenario'])

    summary.to_csv(os.path.join(EMOTION_DIR, f"{model}_{dataset}_summary_scenarios.csv"), index=False)
    pivot = summary.set_index(["effect", "scenario"])[classes]
    return pivot

# ====================== Regression Plots ======================

def plot_heatmap_reg(summary, prefix):
    plt.figure(figsize=(10, 6))
    heatmap_data = summary.pivot_table(index='effect', columns='level', values='mean_valence')
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mean Valence by Effect and Level')
    plt.xlabel('Level')
    plt.ylabel('Effect Name')
    plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_valence_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    heatmap_data = summary.pivot_table(index='effect', columns='level', values='mean_arousal')
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mean Arousal by Effect and Level')
    plt.xlabel('Level')
    plt.ylabel('Effect Name')
    plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_arousal_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_va_trends(summary, prefix):
    plt.figure(figsize=(12, 6))
    for effect in summary['effect'].unique():
        effect_data = summary[summary['effect'] == effect]
        plt.plot(effect_data['level'], effect_data['mean_valence'], label=f'{effect} (Valence)')
        plt.plot(effect_data['level'], effect_data['mean_arousal'], linestyle='--', label=f'{effect} (Arousal)')

    plt.title('Valence and Arousal Trends by Effect and Level')
    plt.xlabel('Level')
    plt.ylabel('Mean Values')
    plt.legend()
    plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_trends.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_va_plane(df, prefix):
    colormap = pyplot.get_cmap('viridis', NUM_LEVEL)

    for effect in df['effect'].unique():
        plt.figure(figsize=(8, 6))
        effect_data = df[df['effect'] == effect]

        for level in sorted(effect_data['level'].unique()):
            level_data = effect_data[effect_data['level'] == level]
            color = colormap((level-1) / max(1, (NUM_LEVEL-1)))
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
        plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_plane_{effect}.png"), dpi=300, bbox_inches='tight')
        plt.close()


def plot_heatmap_reg_scenarios(summary, prefix):
    plt.figure(figsize=(10, 6))
    heatmap_data = summary.pivot_table(index='effect', columns='scenario', values='mean_valence', observed=True)
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mean Valence by Effect and Scenario')
    plt.xlabel('Scenario'); plt.ylabel('Effect Name')
    plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_valence_heatmap_scenarios.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    heatmap_data = summary.pivot_table(index='effect', columns='scenario', values='mean_arousal', observed=True)
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mean Arousal by Effect and Scenario')
    plt.xlabel('Scenario'); plt.ylabel('Effect Name')
    plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_arousal_heatmap_scenarios.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_va_plane_scenarios(df, prefix):
    for effect in df['effect'].unique():
        plt.figure(figsize=(8, 6))
        eff_df = df[df['effect'] == effect]
        for scen in SCENARIO_ORDER:
            sub = eff_df[eff_df['scenario'] == scen]
            if sub.empty:
                continue
            plt.scatter(sub['valence'], sub['arousal'],
                        label=scen, alpha=0.8, s=30, color=SCENARIO_COLORS.get(scen))
        plt.title(f'Valence-Arousal Plane – {effect} (Scenarios)', fontsize=14)
        plt.xlabel('Valence'); plt.ylabel('Arousal')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.7)
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.7)
        plt.grid(alpha=0.3); plt.legend(title='Scenario', fontsize=10)
        plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_plane_{effect}_scenarios.png"), dpi=300, bbox_inches='tight')
        plt.close()


# ====================== Classification Plots ======================

def plot_label_proportions(pivot, prefix):
    effect_list = pivot.index.get_level_values('effect').unique()

    for effect in effect_list:
        data = pivot.loc[effect]
        data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.title(f'Label Proportions for {effect}')
        plt.xlabel('Level')
        plt.ylabel('Proportion')
        plt.legend(title='Label')
        plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_label_proportions_{effect}.png"), dpi=300, bbox_inches='tight')
        plt.close()


def radar_chart(df, title, filename, classes=None):
    df = df.copy()
    if classes is not None:
        # reindex columns in fixed order
        df = df.reindex(columns=classes)

    labels = list(df.columns)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for lvl, row in df.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Level {lvl}')
        ax.fill(angles, values, alpha=0.25)
    ax.set_title(title, size=16, pad=30)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, rotation=20, ha="center")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar_charts(pivot, prefix, classes=None):
    for effect in pivot.index.get_level_values('effect').unique():
        radar_chart(pivot.loc[effect], f'Radar Chart for {effect}', os.path.join(EMOTION_DIR, f"{prefix}_radar_{effect}.png"), classes=classes)


def plot_label_proportions_scenarios(pivot, prefix):
    WRAP_MAP = {"Rage Against The Machine": "Rage Against\nThe Machine"}

    for effect in pivot.index.get_level_values('effect').unique():
        data = pivot.loc[effect]  # rows: scenarios; cols: labels
        # ensure scenario order
        data = data.reindex(SCENARIO_ORDER, axis=0, fill_value=0)

        # wrap long names to two lines
        data = data.copy()
        data.index = [WRAP_MAP.get(ix, ix) for ix in data.index]

        ax = data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        ax.set_title(f'Label Proportions by Scenario – {effect}')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Proportion')
        ax.legend(title='Label', bbox_to_anchor=(1.02, 1), loc='upper left')

        # rotate tick labels diagonally
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment('right')

        plt.tight_layout()
        plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_label_props_{effect}_scenarios.png"),
                    dpi=300)
        plt.close()



def plot_radar_charts_scenarios(pivot, prefix, classes=None):
    for effect in pivot.index.get_level_values('effect').unique():
        df_eff = pivot.loc[effect]
        # keep only the scenarios we have and in fixed order
        df_eff = df_eff.reindex(SCENARIO_ORDER).dropna(how='all')
        radar_chart(df_eff, f'Radar (Scenarios) – {effect}',
                    os.path.join(EMOTION_DIR, f"{prefix}_radar_{effect}_scenarios.png"),
                    classes=classes)

# ====================== Correlation & Chi-Square ======================

def correlation_matrix(df, prefix):
    correlation = df[['level', 'valence', 'arousal']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap: Level, Valence, and Arousal')
    plt.savefig(os.path.join(EMOTION_DIR, f"{prefix}_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()


def chi_square(df, prefix):
    # Chi-square test for association between effect level and labels
    results = []
    for effect in df['effect'].unique():
        contingency_table = pd.crosstab(df[df['effect'] == effect]['level'],
                                         df[df['effect'] == effect]['results'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        results.append(f'{effect}: Chi-square = {chi2:.2f}, p-value = {p:.4f}')
    
    with open(os.path.join(EMOTION_DIR, f"{prefix}_chi_square_results.txt"), "w") as f:
        f.write("\n".join(results))


def emotion():
    # Load the datasets
    deam_mert = pd.read_pickle(os.path.join(DATA_DIR, "deam_mert_results_fx.pkl"))
    deam_clap = pd.read_pickle(os.path.join(DATA_DIR, "deam_clap_results_fx.pkl"))
    deam_qwen = pd.read_pickle(os.path.join(DATA_DIR, "deam_qwen_results_fx.pkl"))
    emopia_mert = pd.read_pickle(os.path.join(DATA_DIR, "emopia_mert_results_fx.pkl"))
    emopia_clap = pd.read_pickle(os.path.join(DATA_DIR, "emopia_clap_results_fx.pkl"))
    emopia_qwen = pd.read_pickle(os.path.join(DATA_DIR, "emopia_qwen_results_fx.pkl"))
    wtf_va_mert = pd.read_pickle(os.path.join(DATA_DIR, "wtf_va_mert_results_fx.pkl"))
    wtf_va_clap = pd.read_pickle(os.path.join(DATA_DIR, "wtf_va_clap_results_fx.pkl"))
    wtf_va_qwen = pd.read_pickle(os.path.join(DATA_DIR, "wtf_va_qwen_results_fx.pkl"))
    wtf_lb_mert = pd.read_pickle(os.path.join(DATA_DIR, "wtf_lb_mert_results_fx.pkl"))
    wtf_lb_clap = pd.read_pickle(os.path.join(DATA_DIR, "wtf_lb_clap_results_fx.pkl"))
    wtf_lb_qwen = pd.read_pickle(os.path.join(DATA_DIR, "wtf_lb_qwen_results_fx.pkl"))

    # Compute statistics
    deam_mert_sum = compute_stats_reg(deam_mert, "mert", "deam")
    deam_clap_sum = compute_stats_reg(deam_clap, "clap", "deam")
    deam_qwen_sum = compute_stats_reg(deam_qwen, "qwen", "deam")

    emopia_mert_norm = compute_stats_clf(emopia_mert, "mert", "emopia")
    emopia_clap_norm = compute_stats_clf(emopia_clap, "clap", "emopia")
    emopia_qwen_norm = compute_stats_clf(emopia_qwen, "qwen", "emopia")

    wtf_va_mert_sum = compute_stats_reg(wtf_va_mert, "mert", "wtf_va")
    wtf_va_clap_sum = compute_stats_reg(wtf_va_clap, "clap", "wtf_va")
    wtf_va_qwen_sum = compute_stats_reg(wtf_va_qwen, "qwen", "wtf_va")
    
    wtf_lb_mert_pivot = compute_stats_multilabel(wtf_lb_mert, "mert", "wtf_lb")
    wtf_lb_clap_pivot = compute_stats_multilabel(wtf_lb_clap, "clap", "wtf_lb")
    wtf_lb_qwen_pivot = compute_stats_multilabel(wtf_lb_qwen, "qwen", "wtf_lb")

    # Regression Plots
    plot_heatmap_reg(deam_mert_sum, "mert_deam")
    plot_heatmap_reg(deam_clap_sum, "clap_deam")
    plot_heatmap_reg(deam_qwen_sum, "qwen_deam")
    plot_heatmap_reg(wtf_va_mert_sum, "mert_wtf_va")
    plot_heatmap_reg(wtf_va_clap_sum, "clap_wtf_va")
    plot_heatmap_reg(wtf_va_qwen_sum, "qwen_wtf_va")

    plot_va_trends(deam_mert_sum, "mert_deam")
    plot_va_trends(deam_clap_sum, "clap_deam")
    plot_va_trends(deam_qwen_sum, "qwen_deam")
    plot_va_trends(wtf_va_mert_sum, "mert_wtf_va")
    plot_va_trends(wtf_va_clap_sum, "clap_wtf_va")
    plot_va_trends(wtf_va_qwen_sum, "qwen_wtf_va")

    plot_va_plane(deam_mert, "mert_deam")
    plot_va_plane(deam_clap, "clap_deam")
    plot_va_plane(deam_qwen, "qwen_deam")
    plot_va_plane(wtf_va_mert, "mert_wtf_va")
    plot_va_plane(wtf_va_clap, "clap_wtf_va")
    plot_va_plane(wtf_va_qwen, "qwen_wtf_va")

    # Classification Plots
    plot_label_proportions(emopia_mert_norm, "mert_emopia")
    plot_label_proportions(emopia_clap_norm, "clap_emopia")
    plot_label_proportions(emopia_qwen_norm, "qwen_emopia")

    plot_radar_charts(emopia_mert_norm, "mert_emopia")
    plot_radar_charts(emopia_clap_norm, "clap_emopia")
    plot_radar_charts(emopia_qwen_norm, "qwen_emopia")
    
    plot_label_proportions(wtf_lb_mert_pivot, "mert_wtf_lb")
    plot_label_proportions(wtf_lb_clap_pivot, "clap_wtf_lb")
    plot_label_proportions(wtf_lb_qwen_pivot, "qwen_wtf_lb")

    plot_radar_charts(wtf_lb_mert_pivot, "mert_wtf_lb", classes=WTF_LB_CLASSES)
    plot_radar_charts(wtf_lb_clap_pivot, "clap_wtf_lb", classes=WTF_LB_CLASSES)
    plot_radar_charts(wtf_lb_qwen_pivot, "qwen_wtf_lb", classes=WTF_LB_CLASSES)

    # Correlation Analysis
    correlation_matrix(deam_mert, "mert_deam")
    correlation_matrix(deam_clap, "clap_deam")
    correlation_matrix(deam_qwen, "qwen_deam")
    correlation_matrix(wtf_va_mert, "mert_wtf_va")
    correlation_matrix(wtf_va_clap, "clap_wtf_va")
    correlation_matrix(wtf_va_qwen, "qwen_wtf_va")

    chi_square(emopia_mert, "mert_emopia")
    chi_square(emopia_clap, "clap_emopia")
    chi_square(emopia_qwen, "qwen_emopia")
    
    print("Visualisation complete!\n")
    
    # -------- Scenario Section (4 cases: Original, Pink Floyd, RATM, U2) --------
    # Load scenario datasets
    wtf_va_mert_s = pd.read_pickle(os.path.join(DATA_DIR, "wtf_va_mert_results_fx_scenarios.pkl"))
    wtf_va_clap_s = pd.read_pickle(os.path.join(DATA_DIR, "wtf_va_clap_results_fx_scenarios.pkl"))
    wtf_va_qwen_s = pd.read_pickle(os.path.join(DATA_DIR, "wtf_va_qwen_results_fx_scenarios.pkl"))
    wtf_lb_mert_s = pd.read_pickle(os.path.join(DATA_DIR, "wtf_lb_mert_results_fx_scenarios.pkl"))
    wtf_lb_clap_s = pd.read_pickle(os.path.join(DATA_DIR, "wtf_lb_clap_results_fx_scenarios.pkl"))
    wtf_lb_qwen_s = pd.read_pickle(os.path.join(DATA_DIR, "wtf_lb_qwen_results_fx_scenarios.pkl"))

    # Compute statistics (scenarios)
    wtf_va_mert_sum_s = compute_stats_reg_scenarios(wtf_va_mert_s, "mert", "wtf_va")
    wtf_va_clap_sum_s = compute_stats_reg_scenarios(wtf_va_clap_s, "clap", "wtf_va")
    wtf_va_qwen_sum_s = compute_stats_reg_scenarios(wtf_va_qwen_s, "qwen", "wtf_va")

    wtf_lb_mert_pivot_s = compute_stats_multilabel_scenarios(wtf_lb_mert_s, "mert", "wtf_lb")
    wtf_lb_clap_pivot_s = compute_stats_multilabel_scenarios(wtf_lb_clap_s, "clap", "wtf_lb")
    wtf_lb_qwen_pivot_s = compute_stats_multilabel_scenarios(wtf_lb_qwen_s, "qwen", "wtf_lb")

    # Regression Plots (scenarios)
    plot_heatmap_reg_scenarios(wtf_va_mert_sum_s, "mert_wtf_va")
    plot_heatmap_reg_scenarios(wtf_va_clap_sum_s, "clap_wtf_va")
    plot_heatmap_reg_scenarios(wtf_va_qwen_sum_s, "qwen_wtf_va")

    plot_va_plane_scenarios(wtf_va_mert_s, "mert_wtf_va")
    plot_va_plane_scenarios(wtf_va_clap_s, "clap_wtf_va")
    plot_va_plane_scenarios(wtf_va_qwen_s, "qwen_wtf_va")

    # Classification Plots (scenarios)
    plot_label_proportions_scenarios(wtf_lb_mert_pivot_s, "mert_wtf_lb")
    plot_label_proportions_scenarios(wtf_lb_clap_pivot_s, "clap_wtf_lb")
    plot_label_proportions_scenarios(wtf_lb_qwen_pivot_s, "qwen_wtf_lb")

    plot_radar_charts_scenarios(wtf_lb_mert_pivot_s, "mert_wtf_lb", classes=WTF_LB_CLASSES)
    plot_radar_charts_scenarios(wtf_lb_clap_pivot_s, "clap_wtf_lb", classes=WTF_LB_CLASSES)
    plot_radar_charts_scenarios(wtf_lb_qwen_pivot_s, "qwen_wtf_lb", classes=WTF_LB_CLASSES)
    print("Scenario visualisation complete!\n")


emotion()
