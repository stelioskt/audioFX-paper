import os, warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning

import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from dotenv import load_dotenv

# ===================== Config ========================

load_dotenv()
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DATA_DIR   = os.getenv("DATA_DIR")
EMOPIA_DIR = os.getenv("EMOPIA_DIR")
WTF_DIR    = os.getenv("WITHEFLOW_DIR")
EMBED_DIR  = os.path.join(OUTPUT_DIR, "embedding")

DATASETS  = ["deam", "emopia", "wtf_va", "wtf_lb"]
MODELS    = ["mert", "clap", "qwen"]
EFFECTS   = ["reverb", "delay", "distortion", "eq", "chorus", "phaser"]
SCENARIOS = ["Pink Floyd", "Rage Against The Machine", "U2"]

EMOPIA_SHAPES = {"Excitement": "o", "Anger": "s", "Sadness": "^", "Calmness": "D"}
WTF_LB_CLASSES = [
    "sadness", "nostalgia", "peacefulness", "neutral",
    "tenderness", "joyfulActivation", "wonder", "transcendence",
    "power", "tension"
]
WTF_LB_SHAPES = {
    "sadness": "^", "nostalgia": "P", "peacefulness": "v", "neutral": "X",
    "tenderness": ">", "joyfulActivation": "o", "wonder": "*", "transcendence": "s",
    "power": "D", "tension": "<"
}

# ============================ I/O =============================

def primary_label(s: str) -> str:
    labs = [x.strip() for x in str(s).split(";") if x.strip()]
    for cls in WTF_LB_CLASSES:
        if cls in labs:
            return cls
    raise ValueError(f"No primary label found in: {s}")

def load_df(dataset: str, model: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{dataset}_{model}_features_fx.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_pickle(path)

    for col in ["features", "effect", "level", "path"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")

    df["features"] = df["features"].apply(lambda x: np.asarray(x, dtype=float))
    df["effect"] = df["effect"].astype(str)

    # attach labels for classification datasets
    if dataset == "emopia":
        csv = pd.read_csv(os.path.join(EMOPIA_DIR, "emopia.csv"))
        lab_map = csv.set_index("path")["emo_class"]
        df["label"] = df["path"].map(lab_map).astype(str).str.strip()
    elif dataset == "wtf_lb":
        csv = pd.read_csv(os.path.join(WTF_DIR, "wtf_lb.csv"))
        lab_map = csv.set_index("path")["labels"].apply(primary_label)
        df["label"] = df["path"].map(lab_map).astype(str).str.strip()
    return df

# ========================== Sampling =============================

def sample_tracks_reg(df_eff: pd.DataFrame, n: int, rng) -> List[str]:
    tracks = df_eff["path"].unique().tolist()
    return rng.choice(tracks, size=min(n, len(tracks)), replace=False).tolist()

def sample_tracks_clf(df_eff: pd.DataFrame, per_label: int, rng) -> List[str]:
    if "label" not in df_eff.columns:
        raise ValueError("classification PKLs must include a 'label' column.")
    selected = []
    for _, grp in df_eff.groupby("label"):
        tracks = grp["path"].unique().tolist()
        k = min(per_label, len(tracks))
        if k > 0:
            selected.extend(rng.choice(tracks, size=k, replace=False).tolist())
    return selected

def sample_tracks_clf_with_labels(df_eff: pd.DataFrame, per_label: int, labels_subset: List[str], rng) -> List[str]:
    selected = []
    for lab in labels_subset:
        grp = df_eff[df_eff["label"] == lab]
        if grp.empty: 
            continue
        tracks = grp["path"].unique().tolist()
        k = min(per_label, len(tracks))
        if k > 0:
            selected.extend(rng.choice(tracks, size=k, replace=False).tolist())
    return selected

# ======================= Per-track deltas (Optional) =======================

def make_deltas(df_sub: pd.DataFrame) -> np.ndarray:
    X = np.vstack(df_sub["features"].values)
    paths = df_sub["path"].values
    levels = df_sub["level"].values
    X_delta = np.empty_like(X)
    for t in np.unique(paths):
        idx = np.where(tracks := (paths == t))[0]
        b = idx[np.argmin(levels[idx])]
        baseline = X[b]
        X_delta[idx] = X[idx] - baseline
    return X_delta

# ==================== Cleaning =======================

def fit_cleaning_state(X: np.ndarray, var_thresh: float = 1e-6, corr_thresh: float = 0.95):
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    vt = VarianceThreshold(threshold=var_thresh).fit(Xz)
    X1 = vt.transform(Xz)

    if X1.shape[1] <= 1:
        keep_corr = np.arange(X1.shape[1], dtype=int)
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
        keep_corr = np.array([i for i in range(n) if i not in to_drop], dtype=int)

    return {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "vt_support": vt.get_support(indices=True),
        "keep_corr": keep_corr,
    }

def apply_cleaning(X: np.ndarray, state: Dict[str, np.ndarray]) -> np.ndarray:
    Xz = (X - state["scaler_mean"]) / state["scaler_scale"]
    X1 = Xz[:, state["vt_support"]]
    return X1[:, state["keep_corr"]]

# ===================== Feature selection =====================

def select_features_reg(Xc: np.ndarray, y: np.ndarray, top_k: int, rng) -> Tuple[np.ndarray, int]:
    if Xc.shape[1] <= top_k:
        return np.arange(Xc.shape[1]), Xc.shape[1]

    enet = ElasticNetCV(
        l1_ratio=[0.5, 0.8],
        alphas=np.logspace(-3, 2, 50),
        cv=5, max_iter=60000, tol=2e-3,
        random_state=rng, verbose=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        enet.fit(Xc, y.astype(float))

    coef = np.abs(enet.coef_)
    nnz = int(np.count_nonzero(coef))
    nz = np.where(coef > 0)[0]
    if len(nz) >= top_k:
        idx = np.argsort(coef)[::-1][:top_k]
    else:
        remaining = np.setdiff1d(np.arange(Xc.shape[1]), nz, assume_unique=True)
        fill = remaining[np.argsort(Xc[:, remaining].var(axis=0))[::-1]][: max(0, top_k - len(nz))]
        idx = np.concatenate([nz, fill])
    return idx, nnz

def select_features_clf(Xc: np.ndarray, y: np.ndarray, top_k: int, rng) -> Tuple[np.ndarray, int]:
    if Xc.shape[1] <= top_k:
        return np.arange(Xc.shape[1]), Xc.shape[1]

    clf = LogisticRegression(
        penalty="elasticnet", l1_ratio=0.5, solver="saga",
        C=0.5, tol=2e-3, max_iter=6000,
        random_state=rng, verbose=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Xc, y)

    coef = np.abs(clf.coef_)
    nnz_total = int((coef > 0).any(axis=0).sum())
    imp = coef.mean(axis=0)
    if not np.any(imp):
        imp = Xc.var(axis=0)
    idx = np.argsort(imp)[::-1][:top_k]
    return idx, nnz_total

def select_top_variance(Xc: np.ndarray, top_k: int) -> np.ndarray:
    var = Xc.var(axis=0)
    k = min(top_k, Xc.shape[1])
    return np.argsort(var)[::-1][:k]

# ========================= UMAP factory tuned by top_k =========================

def make_umap_for(top_k: int, seed: int):
    if top_k <= 32:
        n_neighbors, min_dist = 30, 0.5
    elif top_k <= 64:
        n_neighbors, min_dist = 20, 0.3
    else:
        n_neighbors, min_dist = 15, 0.2
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        init="spectral",
        random_state=seed,
        n_jobs=1
    )

# ======================== Plotting ========================

def plot_umap_trajectories(
    df_sub: pd.DataFrame, X_sel: np.ndarray, title: str, ax: plt.Axes,
    label_mode: Optional[str], seed: int, top_k: int, shape_map: Optional[Dict[str, str]]=None
):
    reducer = make_umap_for(top_k, seed)
    coords = reducer.fit_transform(X_sel)

    tracks = df_sub["path"].values
    levels = df_sub["level"].values
    labels = df_sub["label"].values if ("label" in df_sub.columns) else None

    norm = plt.Normalize(1, 10)
    cmap = cm.coolwarm
    last_scatter = None

    for t in np.unique(tracks):
        idx = np.where(tracks == t)[0]
        order = np.argsort(levels[idx])
        pts = coords[idx[order]]
        lev = levels[idx[order]]

        ax.scatter(pts[0, 0], pts[0, 1], marker="x", c="black", s=40)
        marker = "o"
        if (label_mode == "shape") and (labels is not None):
            lab0 = labels[idx[order]][0]
            marker = shape_map.get(lab0, "o") if shape_map is not None else "o"

        if pts.shape[0] > 1:
            last_scatter = ax.scatter(
                pts[1:, 0], pts[1:, 1],
                c=lev[1:], cmap=cmap, norm=norm, s=24, marker=marker,
                edgecolors="black", linewidths=0.2
            )
            ax.plot(pts[:, 0], pts[:, 1], color="gray", alpha=0.35, linewidth=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return last_scatter

# ========================== Pipeline ==========================

def embedding(top_k, n_tracks_deam, n_tracks_emopia_per_label, seed, global_feature_selection, use_deltas):
    rng = check_random_state(seed)
    os.makedirs(EMBED_DIR, exist_ok=True)

    dfs: Dict[Tuple[str, str], pd.DataFrame] = {(ds, m): load_df(ds, m) for ds in DATASETS for m in MODELS}

    # Fit cleaning per (dataset, model)
    clean_state: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for ds in DATASETS:
        for m in MODELS:
            df_all = dfs[(ds, m)]
            X_all = make_deltas(df_all) if use_deltas else np.vstack(df_all["features"].values)
            state = fit_cleaning_state(X_all, var_thresh=1e-6, corr_thresh=0.95)
            Xc_all = apply_cleaning(X_all, state)
            clean_state[(ds, m)] = state
            print(f"[{ds}|{m}] cleaned dims: {X_all.shape[1]} -> {Xc_all.shape[1]}")

    # Global feature selection in the same cleaned space
    global_sel: Dict[Tuple[str, str], Tuple[np.ndarray, int]] = {}
    if global_feature_selection:
        print("\n=== Global feature selection (per dataset, model) ===")
        for ds in DATASETS:
            for m in MODELS:
                df_all = dfs[(ds, m)]
                state = clean_state[(ds, m)]
                X_all = make_deltas(df_all) if use_deltas else np.vstack(df_all["features"].values)
                Xc_all = apply_cleaning(X_all, state)
                if ds in {"deam", "wtf_va"}:
                    idx, nnz = select_features_reg(Xc_all, df_all["level"].values.astype(float), top_k, rng)
                else:
                    idx, nnz = select_features_clf(Xc_all, df_all["label"].values, top_k, rng)
                global_sel[(ds, m)] = (idx, nnz)

    # ---------------- DEAM ----------------
    print("\n=== DEAM ===")
    fig_deam, axes_deam = plt.subplots(len(MODELS), len(EFFECTS),
                                       figsize=(4*len(EFFECTS), 4*len(MODELS)))
    axes_deam = np.atleast_2d(axes_deam)
    last_sc_deam = None

    for r, model in enumerate(MODELS):
        df_m = dfs[("deam", model)]
        state = clean_state[("deam", model)]
        for c, effect in enumerate(EFFECTS):
            ax = axes_deam[r, c]
            df_eff = df_m[df_m["effect"] == effect]
            if df_eff.empty:
                ax.set_title(f"DEAM – {model.upper()} – {effect}\n(no rows)")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            tracks = sample_tracks_reg(df_eff, n_tracks_deam, rng)
            df_sub = df_eff[df_eff["path"].isin(tracks)].copy()

            X = make_deltas(df_sub) if use_deltas else np.vstack(df_sub["features"].values)
            Xc = apply_cleaning(X, state)

            if global_feature_selection:
                idx, nnz = global_sel[("deam", model)]
            else:
                idx, nnz = select_features_reg(Xc, df_sub["level"].values.astype(float), top_k, rng)

            X_sel = Xc[:, idx[: min(top_k, Xc.shape[1])]]
            title = f"DEAM – {model.upper()} – {effect}"
            last_sc_deam = plot_umap_trajectories(
                df_sub, X_sel, title, ax=ax, label_mode=None, seed=seed, top_k=top_k
            )

    if last_sc_deam is not None:
        fig_deam.subplots_adjust(left=0.10, right=0.98, wspace=0.25, hspace=0.25)
        cax = fig_deam.add_axes([0.03, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(1, 10))
        fig_deam.colorbar(sm, cax=cax, label="Effect Level")
    fig_deam.suptitle("DEAM – UMAP Trajectories (ElasticNet-selected top-K features)", fontsize=14)
    out_deam = os.path.join(EMBED_DIR, "deam_umap_grid.png")
    fig_deam.savefig(out_deam, dpi=200)

    # ---------------- WTF_VA (regression) ----------------
    print("\n=== WTF_VA ===")
    fig_wtfva, axes_wtfva = plt.subplots(len(MODELS), len(EFFECTS),
                                         figsize=(4*len(EFFECTS), 4*len(MODELS)))
    axes_wtfva = np.atleast_2d(axes_wtfva)
    last_sc_wtfva = None

    for r, model in enumerate(MODELS):
        df_m = dfs[("wtf_va", model)]
        state = clean_state[("wtf_va", model)]
        for c, effect in enumerate(EFFECTS):
            ax = axes_wtfva[r, c]
            df_eff = df_m[df_m["effect"] == effect]
            if df_eff.empty:
                ax.set_title(f"WTF_VA – {model.upper()} – {effect}\n(no rows)")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            tracks = sample_tracks_reg(df_eff, n_tracks_deam, rng)
            df_sub = df_eff[df_eff["path"].isin(tracks)].copy()

            X = make_deltas(df_sub) if use_deltas else np.vstack(df_sub["features"].values)
            Xc = apply_cleaning(X, state)

            if global_feature_selection:
                idx, nnz = global_sel[("wtf_va", model)]
            else:
                idx, nnz = select_features_reg(Xc, df_sub["level"].values.astype(float), top_k, rng)

            X_sel = Xc[:, idx[: min(top_k, Xc.shape[1])]]
            title = f"WTF_VA – {model.upper()} – {effect}"
            last_sc_wtfva = plot_umap_trajectories(
                df_sub, X_sel, title, ax=ax, label_mode=None, seed=seed, top_k=top_k
            )

    if last_sc_wtfva is not None:
        fig_wtfva.subplots_adjust(left=0.10, right=0.98, wspace=0.25, hspace=0.25)
        cax = fig_wtfva.add_axes([0.03, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(1, 10))
        fig_wtfva.colorbar(sm, cax=cax, label="Effect Level")
    fig_wtfva.suptitle("WTF_VA – UMAP Trajectories (ElasticNet-selected top-K)", fontsize=14)
    out_wtfva = os.path.join(EMBED_DIR, "wtf_va_umap_grid.png")
    fig_wtfva.savefig(out_wtfva, dpi=200)

    # ---------------- EMOPIA ----------------
    print("\n=== EMOPIA ===")
    fig_emo, axes_emo = plt.subplots(len(MODELS), len(EFFECTS),
                                     figsize=(4*len(EFFECTS), 4*len(MODELS)))
    axes_emo = np.atleast_2d(axes_emo)
    last_sc_emo = None

    for r, model in enumerate(MODELS):
        df_m = dfs[("emopia", model)]
        if "label" not in df_m.columns:
            raise ValueError("EMOPIA PKLs must include a 'label' column.")
        state = clean_state[("emopia", model)]
        for c, effect in enumerate(EFFECTS):
            ax = axes_emo[r, c]
            df_eff = df_m[df_m["effect"] == effect]
            if df_eff.empty:
                ax.set_title(f"EMOPIA – {model.upper()} – {effect}\n(no rows)")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            tracks = sample_tracks_clf(df_eff, n_tracks_emopia_per_label, rng)
            df_sub = df_eff[df_eff["path"].isin(tracks)].copy()

            X = make_deltas(df_sub) if use_deltas else np.vstack(df_sub["features"].values)
            Xc = apply_cleaning(X, state)

            if global_feature_selection:
                idx, nnz = global_sel[("emopia", model)]
            else:
                idx, nnz = select_features_clf(Xc, df_sub["label"].values, top_k, rng)

            X_sel = Xc[:, idx[: min(top_k, Xc.shape[1])]]
            title = f"EMOPIA – {model.upper()} – {effect}"
            last_sc_emo = plot_umap_trajectories(
                df_sub, X_sel, title, ax=ax, label_mode="shape", seed=seed, top_k=top_k, shape_map=EMOPIA_SHAPES
            )

    if last_sc_emo is not None:
        fig_emo.subplots_adjust(left=0.10, right=0.98, wspace=0.25, hspace=0.25)
        cax = fig_emo.add_axes([0.03, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(1, 10))
        fig_emo.colorbar(sm, cax=cax, label="Effect Level")

        legend_elems = [Line2D([0],[0], marker=v, color='w', markerfacecolor='gray',
                               markersize=7, linestyle='None', label=k)
                        for k, v in EMOPIA_SHAPES.items()]
        fig_emo.legend(handles=legend_elems, loc="upper center",
                       bbox_to_anchor=(0.5, 1.06), ncol=4, title="EMOPIA Labels", frameon=False)

    fig_emo.suptitle("EMOPIA – UMAP Trajectories (ElasticNet/Logistic-ENet top-K)", fontsize=14)
    out_emo = os.path.join(EMBED_DIR, "emopia_umap_grid.png")
    fig_emo.savefig(out_emo, dpi=200)

    # ---------------------- WTF_LB (multilabel) ----------------------
    print("\n=== WTF_LB ===")
    fig_wtflb, axes_wtflb = plt.subplots(len(MODELS), len(EFFECTS),
                                         figsize=(4*len(EFFECTS), 4*len(MODELS)))
    axes_wtflb = np.atleast_2d(axes_wtflb)
    last_sc_wtflb = None

    for r, model in enumerate(MODELS):
        df_m = dfs[("wtf_lb", model)]
        if "label" not in df_m.columns:
            raise ValueError("WTF_LB PKLs must include a 'label' column.")
        state = clean_state[("wtf_lb", model)]
        for c, effect in enumerate(EFFECTS):
            ax = axes_wtflb[r, c]
            df_eff = df_m[df_m["effect"] == effect]
            if df_eff.empty:
                ax.set_title(f"WTF_LB – {model.upper()} – {effect}\n(no rows)")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            tracks = sample_tracks_clf(df_eff, n_tracks_emopia_per_label, rng)
            df_sub = df_eff[df_eff["path"].isin(tracks)].copy()

            X = make_deltas(df_sub) if use_deltas else np.vstack(df_sub["features"].values)
            Xc = apply_cleaning(X, state)

            if global_feature_selection:
                idx, nnz = global_sel[("wtf_lb", model)]
            else:
                idx, nnz = select_features_clf(Xc, df_sub["label"].values, top_k, rng)

            X_sel = Xc[:, idx[: min(top_k, Xc.shape[1])]]
            title = f"WTF_LB – {model.upper()} – {effect}"
            last_sc_wtflb = plot_umap_trajectories(
                df_sub, X_sel, title, ax=ax, label_mode="shape", seed=seed, top_k=top_k, shape_map=WTF_LB_SHAPES
            )

    if last_sc_wtflb is not None:
        fig_wtflb.subplots_adjust(left=0.10, right=0.98, wspace=0.25, hspace=0.25)
        cax = fig_wtflb.add_axes([0.03, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(1, 10))
        fig_wtflb.colorbar(sm, cax=cax, label="Effect Level")

        legend_elems = [Line2D([0],[0], marker=v, color='w', markerfacecolor='gray',
                               markersize=7, linestyle='None', label=k)
                        for k, v in WTF_LB_SHAPES.items()]
        fig_wtflb.legend(handles=legend_elems, loc="upper center",
                         bbox_to_anchor=(0.5, 1.06), ncol=5, title="WTF_LB Primary Labels", frameon=False)

    fig_wtflb.suptitle("WTF_LB – UMAP Trajectories (Logistic-ENet top-K; primary label)", fontsize=14)
    out_wtflb = os.path.join(EMBED_DIR, "wtf_lb_umap_grid.png")
    fig_wtflb.savefig(out_wtflb, dpi=200)

    print(f"\n[OK] Saved: {out_deam}")
    print(f"[OK] Saved: {out_wtfva}")
    print(f"[OK] Saved: {out_emo}")
    print(f"[OK] Saved: {out_wtflb}")

# ========================== Scenario Tables ==========================

def embedding_scenarios(top_k, seed, n_tracks_wtf_va=20, n_per_label_wtf_lb=3, labels_subset_wtf_lb=None):
    """
    Build 3×3 grids (rows=models, cols=configurations) that overlay Original vs each Band setting.
    Dataset handled: wtf_va (regression, no shapes) and wtf_lb (primary-label shapes).
    """
    os.makedirs(EMBED_DIR, exist_ok=True)
    rng = check_random_state(seed)
    if labels_subset_wtf_lb is None:
        labels_subset_wtf_lb = WTF_LB_LABELS_FOR_SCENARIO

    # ---------- visual style ----------
    plt.rcParams.update({
        "figure.dpi": 200,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.facecolor": "white",
        "axes.edgecolor": "#d0d0d0",
        "axes.grid": True,
        "grid.color": "#e9e9e9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
    scenario_color = {
        "Pink Floyd": "tab:blue",
        "Rage Against The Machine": "tab:orange",
        "U2": "tab:green",
    }
    orig_style = dict(s=46, facecolors="none", edgecolors="#4d4d4d", linewidths=1.0, zorder=2, alpha=0.9)
    band_style = dict(s=54, edgecolors="#1a1a1a", linewidths=0.35, zorder=3, alpha=0.92)

    # --- prepare cleaning states from base features ---
    clean_state: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for dataset in ["wtf_va", "wtf_lb"]:
        for model in MODELS:
            base_path = os.path.join(DATA_DIR, f"{dataset}_{model}_features_fx.pkl")
            if not os.path.exists(base_path):
                continue
            df_all = pd.read_pickle(base_path)
            X_all = np.vstack(df_all["features"].values)
            state = fit_cleaning_state(X_all, var_thresh=1e-6, corr_thresh=0.95)
            clean_state[(dataset, model)] = state

    def _decorate_cell(ax, row_model: str, col_scen: str, is_first_col: bool):
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#d0d0d0")
            ax.spines[side].set_linewidth(0.8)
        ax.set_xticks([]); ax.set_yticks([])
        if col_scen:
            ax.set_title(col_scen, pad=4)
        if is_first_col:
            ax.set_ylabel(row_model.upper(), fontsize=11, fontweight="bold")

    all_legend_handles = (
        Line2D([0],[0], marker="o", linestyle="None",
               markerfacecolor="none", markeredgecolor="#4d4d4d", markersize=6.5, label="Original"),
        Line2D([0],[0], marker="o", linestyle="None",
               markerfacecolor=scenario_color["Pink Floyd"], markeredgecolor="#ffffff", markersize=7.2, label="Pink Floyd"),
        Line2D([0],[0], marker="o", linestyle="None",
               markerfacecolor=scenario_color["Rage Against The Machine"], markeredgecolor="#1a1a1a", markersize=7.2, label="Rage Against The Machine"),
        Line2D([0],[0], marker="o", linestyle="None",
               markerfacecolor=scenario_color["U2"], markeredgecolor="#1a1a1a", markersize=7.2, label="U2"),
    )

    def plot_dataset(dataset: str):
        print(f"\n=== SCENARIOS GRID: {dataset.upper()} ===")
        fig, axes = plt.subplots(
            nrows=len(MODELS), ncols=len(SCENARIOS),
            figsize=(4.8*len(SCENARIOS), 4.6*len(MODELS))
        )
        axes = np.atleast_2d(axes)

        for r, model in enumerate(MODELS):               # rows = models
            scen_path = os.path.join(DATA_DIR, f"{dataset}_{model}_features_fx_scenarios.pkl")
            for c, scen in enumerate(SCENARIOS):
                ax = axes[r, c]
                _decorate_cell(ax, model, scen if r == 0 else "", is_first_col=(c == 0))

            if not os.path.exists(scen_path):
                for c in range(len(SCENARIOS)):
                    ax = axes[r, c]
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=10, color="#666")
                continue

            df_s = pd.read_pickle(scen_path).copy()

            if dataset == "wtf_lb" and "label" not in df_s.columns:
                csv = pd.read_csv(os.path.join(WTF_DIR, "wtf_lb.csv"))
                lab_map = csv.set_index("path")["labels"].apply(primary_label)
                df_s["label"] = df_s["path"].map(lab_map).astype(str).str.strip()

            for c, scen in enumerate(SCENARIOS):         # cols = configurations
                ax = axes[r, c]

                df_orig = df_s[df_s["scenario"] == "Original"].copy()
                df_band = df_s[df_s["scenario"] == scen].copy()
                common_paths = np.intersect1d(df_orig["path"].unique(), df_band["path"].unique())

                if dataset == "wtf_va":
                    if len(common_paths) == 0:
                        ax.text(0.5, 0.5, "no common paths", ha="center", va="center", fontsize=10, color="#666")
                        continue
                    sample = rng.choice(common_paths, size=min(n_tracks_wtf_va, len(common_paths)), replace=False)
                    df_orig_sub = df_orig[df_orig["path"].isin(sample)].copy()
                    df_band_sub = df_band[df_band["path"].isin(sample)].copy()
                else:
                    df_orig = df_orig[df_orig["path"].isin(common_paths)]
                    df_band = df_band[df_band["path"].isin(common_paths)]
                    df_orig = df_orig[df_orig["label"].isin(labels_subset_wtf_lb)]
                    df_band = df_band[df_band["label"].isin(labels_subset_wtf_lb)]

                    tracks_sel = []
                    for lab in labels_subset_wtf_lb:
                        p_orig = set(df_orig[df_orig["label"] == lab]["path"].unique())
                        p_band = set(df_band[df_band["label"] == lab]["path"].unique())
                        inter = list(p_orig.intersection(p_band))
                        if len(inter) > 0:
                            k = min(n_per_label_wtf_lb, len(inter))
                            tracks_sel += rng.choice(inter, size=k, replace=False).tolist()
                    if not tracks_sel:
                        ax.text(0.5, 0.5, "no label-balanced paths", ha="center", va="center", fontsize=10, color="#666")
                        continue
                    df_orig_sub = df_orig[df_orig["path"].isin(tracks_sel)].copy()
                    df_band_sub = df_band[df_band["path"].isin(tracks_sel)].copy()

                # Combine and tag
                df_orig_sub = df_orig_sub.assign(which="Original", level=1)
                df_band_sub = df_band_sub.assign(which=scen,       level=1)
                df_sub = pd.concat([df_orig_sub, df_band_sub], ignore_index=True)

                # Cleaning & selection (local to the cell)
                state = clean_state.get((dataset, model))
                X = np.vstack(df_sub["features"].values)
                Xc = apply_cleaning(X, state)

                if dataset == "wtf_va":
                    idx = select_top_variance(Xc, top_k)
                else:
                    idx, _ = select_features_clf(Xc, df_sub["label"].values, top_k, rng)
                X_sel = Xc[:, idx[: min(top_k, Xc.shape[1])]]

                # UMAP
                reducer = make_umap_for(top_k, seed)
                coords = reducer.fit_transform(X_sel)

                color_band = scenario_color[scen]
                which = df_sub["which"].values

                if dataset == "wtf_va":
                    mask_o = (which == "Original")
                    mask_b = ~mask_o
                    ax.scatter(coords[mask_o, 0], coords[mask_o, 1], **orig_style)
                    ax.scatter(coords[mask_b, 0], coords[mask_b, 1], c=color_band, **band_style)
                else:
                    labels = df_sub["label"].values
                    for i in range(len(df_sub)):
                        mk = WTF_LB_SHAPES.get(labels[i], "o")
                        if which[i] == "Original":
                            ax.scatter(coords[i,0], coords[i,1], marker=mk, **orig_style)
                        else:
                            ax.scatter(coords[i,0], coords[i,1], marker=mk, c=color_band, **band_style)

        fig.suptitle(f"{dataset.upper()} – Scenarios (models × configs)", fontsize=12, fontweight="semibold", y=0.98)

        fig.legend(
            handles=all_legend_handles,
            loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02)
        )

        fig.tight_layout(rect=[0.03, 0.06, 0.997, 0.955])
        fig.subplots_adjust(wspace=0.20, hspace=0.20)

        out = os.path.join(EMBED_DIR, f"{dataset}_scenarios_grid.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"[OK] Saved scenario grid: {out}")

    # build both requested grids
    plot_dataset("wtf_va")
    plot_dataset("wtf_lb")