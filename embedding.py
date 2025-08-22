import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.utils import check_random_state

import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
load_dotenv()
DATASETS  = ["deam", "emopia"]
MODELS    = ["mert", "clap", "qwen"]
EFFECTS   = ["reverb", "delay", "distortion", "eq", "chorus", "phaser"]
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
EMBED_DIR = os.path.join(OUTPUT_DIR, "embedding")
DATA_DIR  = os.getenv("DATA_DIR")
EMOPIA_DIR = os.getenv("EMOPIA_DIR")

# -----------------------------
# I/O
# -----------------------------
def load_df(dataset: str, model: str) -> pd.DataFrame:
    """
    Load one PKL and enforce simple schema & types.
    Required columns: features (array), effect (str), level (int/float), path (str).
    EMOPIA must also have 'label' (str with 4 classes).
    """
    path = os.path.join(DATA_DIR, f"{dataset}_{model}_features_fx.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_pickle(path)

    # basic checks
    for col in ["features", "effect", "level", "path"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")

    # normalize
    df["features"] = df["features"].apply(lambda x: np.asarray(x, dtype=float))
    
    # insert a 'label' column indicating the label of the initial sample (only for emopia)
    if dataset == 'emopia':
        csv = pd.read_csv(os.path.join(EMOPIA_DIR, "emopia.csv"))
        df["label"] = df["path"].map(csv.set_index("path")["emo_class"])
    return df

# -----------------------------
# Sampling (simplified)
# -----------------------------
def sample_tracks_deam(df_eff: pd.DataFrame, n: int, rng) -> List[str]:
    """Pick up to n unique track IDs (paths) for DEAM."""
    tracks = df_eff["path"].unique().tolist()
    k = min(n, len(tracks))
    return rng.choice(tracks, size=k, replace=False).tolist()

def sample_tracks_emopia(df_eff: pd.DataFrame, per_label: int, rng) -> List[str]:
    """Pick up to per_label tracks per EMOPIA class (balanced)."""
    if "label" not in df_eff.columns:
        raise ValueError("EMOPIA PKLs must include a 'label' column (4 classes).")
    selected = []
    for _, grp in df_eff.groupby("label"):
        tracks = grp["path"].unique().tolist()
        k = min(per_label, len(tracks))
        if k > 0:
            selected.extend(rng.choice(tracks, size=k, replace=False).tolist())
    return selected

# -----------------------------
# Feature cleaning (with logging)
# -----------------------------
def clean_features(X: np.ndarray, var_thresh: float = 1e-6, corr_thresh: float = 0.95) -> np.ndarray:
    """
    Standardize -> drop low-variance -> drop highly correlated.
    Returns a cleaned matrix (same row count, fewer columns).
    Prints how many features survive each step.
    """
    D0 = X.shape[1]

    # 1) standardize
    Xz = StandardScaler().fit_transform(X)

    # 2) remove almost-constant features
    vt = VarianceThreshold(threshold=var_thresh)
    X1 = vt.fit_transform(Xz)
    D1 = X1.shape[1]

    # 3) remove one of any pair with |corr| > corr_thresh
    if D1 <= 1:
        X_final = X1
    else:
        C = np.corrcoef(X1, rowvar=False)
        to_drop = set()
        n = C.shape[0]
        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i + 1, n):
                if j in to_drop:
                    continue
                if np.isfinite(C[i, j]) and abs(C[i, j]) > corr_thresh:
                    to_drop.add(j)
        keep = np.array([i for i in range(n) if i not in to_drop], dtype=int)
        X_final = X1[:, keep]

    D2 = X_final.shape[1]
    print(f"    Features: start={D0} | after low-var={D1} | after corr={D2} (dropped {D0-D1} + {D1-D2})")
    return X_final

# ---------------------
# Feature selection
# ---------------------
def select_features_deam(Xc: np.ndarray, y: np.ndarray, top_k: int, rng) -> np.ndarray:
    """
    ElasticNetCV for continuous level; prints chosen alpha and l1_ratio.
    If Elastic Net yields fewer than top_k nonzeros, fill the remainder by variance.
    """
    if Xc.shape[1] <= top_k:
        print(f"    Selection: features <= top_k ({Xc.shape[1]} <= {top_k}), using all.")
        return np.arange(Xc.shape[1])

    enet = ElasticNetCV(
        l1_ratio=[0.2, 0.5, 0.8, 0.95],           # explore sparsity levels
        alphas=np.logspace(-4, 1.5, 60),            # 1e-4 .. 10
        cv=5,
        max_iter=80000,
        tol=1e-3,
        random_state=rng
    )
    enet.fit(Xc, y.astype(float))
    coef = enet.coef_
    nnz = int(np.count_nonzero(coef))
    print(f"    ElasticNet: alpha = {enet.alpha_:.3e} | l1_ratio = {enet.l1_ratio_:.2f} | nonzero = {nnz}/{Xc.shape[1]}")

    # Primary importance from ENet
    imp = np.abs(coef)

    # Hybrid fill-up: if fewer than top_k nonzeros, use high-variance remaining dims
    nonzero_idx = np.where(imp > 0)[0]
    if len(nonzero_idx) >= top_k:
        idx = np.argsort(imp)[::-1][:top_k]
    else:
        remaining = np.setdiff1d(np.arange(Xc.shape[1]), nonzero_idx, assume_unique=True)
        if remaining.size > 0:
            var_rank = remaining[np.argsort(Xc[:, remaining].var(axis=0))[::-1]]
            fill = var_rank[: max(0, top_k - len(nonzero_idx))]
            idx = np.concatenate([nonzero_idx, fill])
        else:
            idx = nonzero_idx  # edge case
    print(f"    Selection: top_k_used={len(idx)} (requested {top_k})")
    return idx

def select_features_emopia(Xc: np.ndarray, y: np.ndarray, top_k: int, rng) -> np.ndarray:
    """Multinomial LogisticRegression L1; prints #non-zero per class."""
    if Xc.shape[1] <= top_k:
        print(f"    Selection: features <= top_k ({Xc.shape[1]} <= {top_k}), using all.")
        return np.arange(Xc.shape[1])

    clf = LogisticRegression(
        penalty="l1", solver="saga", multi_class="multinomial",
        max_iter=6000, C=1.0, random_state=rng
    )
    clf.fit(Xc, y)
    coef = clf.coef_  # shape: (n_classes, n_features)
    nnz_per_class = (np.abs(coef) > 0).sum(axis=1)
    print(f"    Logistic L1: nonzero per class = {nnz_per_class.tolist()} (total feats={Xc.shape[1]})")
    imp = np.abs(coef).mean(axis=0)
    if not np.any(imp):
        imp = Xc.var(axis=0)  # fallback
    idx = np.argsort(imp)[::-1][:top_k]
    print(f"    Selection: top_k={top_k}")
    return idx

# -----------------------------
# Plotting (paper-style)
# -----------------------------
def plot_umap_trajectories(df_sub: pd.DataFrame, X_sel: np.ndarray, title: str, ax: plt.Axes, label_mode: Optional[str] = None, seed: int = 42):
    """
    UMAP(2D) on selected features; draw per-track trajectories ordered by level.
    - first point: black 'x'
    - subsequent points: colored by level (1..10)
    - EMOPIA-only: marker shape encodes label (4 classes)
    """
    reducer = umap.UMAP(n_neighbors=3, random_state=seed, n_jobs=1)
    coords = reducer.fit_transform(X_sel)

    tracks = df_sub["path"].values
    levels = df_sub["level"].values
    labels = df_sub["label"].values if ("label" in df_sub.columns) else None

    norm = plt.Normalize(1, 10)  # fixed color scale
    cmap = cm.coolwarm
    shape_map = {"happy": "o", "sad": "s", "angry": "^", "relaxed": "D"}

    last_scatter = None
    for t in np.unique(tracks):
        idx = np.where(tracks == t)[0]
        order = np.argsort(levels[idx])   # plot in level order
        pts = coords[idx[order]]
        lev = levels[idx[order]]

        # origin
        ax.scatter(pts[0, 0], pts[0, 1], marker="x", c="black", s=40)

        # class-aware marker for EMOPIA
        marker = "o"
        if (label_mode == "shape") and (labels is not None):
            marker = shape_map.get(labels[idx[order]][0], "o")

        # the rest + trajectory line
        if pts.shape[0] > 1:
            last_scatter = ax.scatter(pts[1:, 0], pts[1:, 1], c=lev[1:], cmap=cmap, norm=norm, s=18, marker=marker)
            ax.plot(pts[:, 0], pts[:, 1], color="gray", alpha=0.35, linewidth=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return last_scatter  # for the colorbar

# -----------------------------
# Pipeline
# -----------------------------
def embedding(top_k: int = 64, n_tracks_deam: int = 20, n_tracks_emopia_per_label: int = 5, seed: int = 42):
    rng = check_random_state(seed)
    os.makedirs(EMBED_DIR, exist_ok=True)

    # Load all dataframes once
    dfs: Dict[Tuple[str, str], pd.DataFrame] = {(ds, m): load_df(ds, m) for ds in DATASETS for m in MODELS}

    # ---------- DEAM ----------
    print("\n=== DEAM ===")
    fig_deam, axes_deam = plt.subplots(len(MODELS), len(EFFECTS), figsize=(4*len(EFFECTS), 4*len(MODELS)))
    axes_deam = np.atleast_2d(axes_deam)
    last_sc_deam = None

    for r, model in enumerate(MODELS):
        df_m = dfs[("deam", model)]
        for c, effect in enumerate(EFFECTS):
            ax = axes_deam[r, c]
            df_eff = df_m[df_m["effect"] == effect]
            if df_eff.empty:
                ax.set_title(f"DEAM – {model.upper()} – {effect}\n(no rows)")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            tracks = sample_tracks_deam(df_eff, n_tracks_deam, rng)
            df_sub = df_eff[df_eff["path"].isin(tracks)].copy()
            print(f"- DEAM | {model:<5} | {effect:<10} | tracks={len(tracks):>3} | rows={len(df_sub):>4}")

            X = np.vstack(df_sub["features"].values)
            y = df_sub["level"].values.astype(float)

            Xc = clean_features(X)
            sel = select_features_deam(Xc, y, top_k, rng)
            X_sel = Xc[:, sel]

            title = f"DEAM – {model.upper()} – {effect}"
            last_sc_deam = plot_umap_trajectories(df_sub, X_sel, title, ax=ax, label_mode=None, seed=seed)

    if last_sc_deam is not None:
        # colorbar on the far left
        fig_deam.subplots_adjust(left=0.10, right=0.98, wspace=0.25, hspace=0.25)
        cax = fig_deam.add_axes([0.03, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(1, 10))
        fig_deam.colorbar(sm, cax=cax, label="Effect Level")
    fig_deam.suptitle("DEAM – UMAP Trajectories (ElasticNet-selected top-K features)", fontsize=14)
    fig_deam.tight_layout(rect=[0.12, 0.02, 0.98, 0.95])
    out_deam = os.path.join(EMBED_DIR, "deam_umap_grid.png")
    fig_deam.savefig(out_deam, dpi=200)

    # ---------- EMOPIA ----------
    print("\n=== EMOPIA ===")
    fig_emo, axes_emo = plt.subplots(len(MODELS), len(EFFECTS), figsize=(4*len(EFFECTS), 4*len(MODELS)))
    axes_emo = np.atleast_2d(axes_emo)
    last_sc_emo = None

    for r, model in enumerate(MODELS):
        df_m = dfs[("emopia", model)]
        if "label" not in df_m.columns:
            raise ValueError("EMOPIA PKLs must include a 'label' column with 4 classes.")
        for c, effect in enumerate(EFFECTS):
            ax = axes_emo[r, c]
            df_eff = df_m[df_m["effect"] == effect]
            if df_eff.empty:
                ax.set_title(f"EMOPIA – {model.upper()} – {effect}\n(no rows)")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            tracks = sample_tracks_emopia(df_eff, n_tracks_emopia_per_label, rng)
            df_sub = df_eff[df_eff["path"].isin(tracks)].copy()
            lbl_counts = df_sub.groupby("label")["path"].nunique().to_dict()
            print(f"- EMOPIA | {model:<5} | {effect:<10} | tracks={len(tracks):>3} | rows={len(df_sub):>4} | per-label={lbl_counts}")

            X = np.vstack(df_sub["features"].values)
            y = df_sub["label"].values

            Xc = clean_features(X)
            sel = select_features_emopia(Xc, y, top_k, rng)
            X_sel = Xc[:, sel]

            title = f"EMOPIA – {model.upper()} – {effect}"
            last_sc_emo = plot_umap_trajectories(df_sub, X_sel, title, ax=ax, label_mode="shape", seed=seed)

    if last_sc_emo is not None:
        fig_emo.subplots_adjust(left=0.10, right=0.98, wspace=0.25, hspace=0.25)
        cax = fig_emo.add_axes([0.03, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(1, 10))
        fig_emo.colorbar(sm, cax=cax, label="Effect Level")

        # EMOPIA label legend (shapes)
        shape_map = {"Excitement": "o", "Anger": "s", "Sadness": "^", "Calmness": "D"}
        legend_elems = [
            Line2D([0],[0], marker=v, color='w', markerfacecolor='gray', markersize=7, linestyle='None', label=k)
            for k, v in shape_map.items()
        ]
        fig_emo.legend(handles=legend_elems, loc="upper center", ncol=4, title="EMOPIA Labels")

    fig_emo.suptitle("EMOPIA – UMAP Trajectories (L1-selected top-K features)", fontsize=14)
    fig_emo.tight_layout(rect=[0.12, 0.05, 0.98, 0.95])
    out_emo = os.path.join(EMBED_DIR, "emopia_umap_grid.png")
    fig_emo.savefig(out_emo, dpi=200)

    print(f"\n[OK] Saved: {out_deam}")
    print(f"[OK] Saved: {out_emo}")

# -----------------------------
if __name__ == "__main__":
    embedding(
        top_k=25,
        n_tracks_deam=20,
        n_tracks_emopia_per_label=5,
        seed=42
    )
