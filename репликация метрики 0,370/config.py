"""
WB Hackathon — Centralized Configuration
=========================================
Single source of truth for all pipeline hyperparameters and paths.
Import in any script:  from config import CFG, set_global_seed
"""

import os
import random
from dataclasses import dataclass, field
from typing import List


def set_global_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch (if available)."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[config] Global seed set to {seed}")


@dataclass
class Config:
    """All project-wide settings in one place."""

    # ── Reproducibility ───────────────────────────────────────────────────
    seed: int = 42

    # ── Paths ──────────────────────────────────────────────────────────────
    train_path: str = "train.parquet"
    test_path: str = "test.parquet"
    sample_submission_path: str = "sample_submission.csv"
    google_trends_path: str = "google_trends_daily.csv"

    # Output paths
    tabular_model_dir: str = "models_tabular"
    timeseries_model_dir: str = "models_timeseries"
    submission_tabular: str = "submission_tabular.csv"
    submission_timeseries: str = "submission_timeseries.csv"
    submission_final: str = "submission_final.csv"
    ensemble_details: str = "ensemble_details.csv"

    # ── Validation ─────────────────────────────────────────────────────────
    val_days: int = 14  # Last N days of train = holdout for threshold optimisation
    tuning_days: int = 14  # Days just before holdout = tuning_data for fit()
    prediction_length: int = 14  # Test prediction horizon

    # ── Metric ─────────────────────────────────────────────────────────────
    wmae_weight_positive: float = 7.0  # Weight for days with qty > 0
    wmae_weight_zero: float = 1.0      # Weight for days with qty = 0
    qty_multiple: int = 3              # qty is always a multiple of this

    # ── Tabular pipeline (train_tabular.py) ────────────────────────────────
    classifier_time_limit: int = 1000   # seconds
    regressor_time_limit: int = 1000    # seconds
    classifier_preset: str = "medium_quality"
    regressor_preset: str = "medium_quality"

    # Threshold search range (not used if fixed thresholds are set)
    threshold_min: float = 0.01
    threshold_max: float = 0.95
    threshold_step: float = 0.01
    
    # Fixed thresholds (used when skip_threshold_optimization=True)
    fixed_global_threshold: float = 0.54
    fixed_grouped_thresholds: dict = field(default_factory=lambda: {
        "very_rare": 0.55,
        "rare": 0.42,
        "medium": 0.63,
        "frequent": 0.65,
        "very_frequent": 0.54,
    })
    skip_threshold_optimization: bool = True  # Skip optimization, use fixed thresholds

    # ── Sample weighting ─────────────────────────────────────────────────
    use_sample_weight: bool = False
    sample_weight_half_life: int = 30  # days (exponential decay)

    # ── Grouped threshold (by item_pct_nonzero buckets) ────────────────
    use_grouped_threshold: bool = True
    grouped_threshold_bins: List[float] = field(default_factory=lambda: [
        0.0, 0.03, 0.08, 0.15, 0.30, 1.0,
    ])
    grouped_threshold_labels: List[str] = field(default_factory=lambda: [
        "very_rare", "rare", "medium", "frequent", "very_frequent",
    ])

    # ── Feature selection ──────────────────────────────────────────────
    use_feature_selection: bool = False
    feature_selection_method: str = "permutation_importance"  # permutation_importance | mutual_info | rfe
    feature_selection_top_k: int | None = 20  # None = use importance_threshold instead
    feature_selection_importance_threshold: float = 0.0  # Keep features with importance >= threshold
    feature_selection_clf_weight: float = 0.5  # Weight for classifier importance (1.0 - weight for regressor)

    # ── Dimensionality reduction ────────────────────────────────────────
    use_dimensionality_reduction: bool = True
    dim_reduction_method: str = "umap"  # umap | pca | autoencoder
    dim_reduction_n_components: int = 32  # Target number of dimensions (used if optimization disabled)
    dim_reduction_n_neighbors: int = 15  # UMAP: number of neighbors
    dim_reduction_min_dist: float = 0.1  # UMAP: minimum distance
    
    # Optimization of dimensionality reduction hyperparameters
    optimize_dim_reduction: bool = True  # Optimize hyperparameters on holdout by wMAE
    dim_reduction_n_components_range: List[int] = field(default_factory=lambda: [16, 24, 32, 48, 64])  # Values to try
    dim_reduction_n_neighbors_range: List[int] = field(default_factory=lambda: [10, 15, 20, 30])  # UMAP only
    dim_reduction_min_dist_range: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])  # UMAP only

    # ── Time series pipeline (train_timeseries.py) ─────────────────────────
    ts_time_limit: int = 3600           # seconds
    use_chronos: bool = True            # Set False for CPU-only mode
    chronos_fine_tune_steps: int = 1000
    chronos_fine_tune_lr: float = 1e-5
    known_covariates: List[str] = field(default_factory=lambda: [
        "price", "is_promo", "prev_leftovers", "sneakers_google_trends",
    ])

    # ── Ensemble (ensemble_predict.py) ─────────────────────────────────────
    ensemble_strategy: str = "classifier_gated"  # weighted_average | classifier_gated | max_vote
    ensemble_alpha: float = 0.5  # weight for tabular in blend (0 = all TS, 1 = all tabular)
    ensemble_threshold: float | None = None  # None = use threshold from tabular config

    # ── Google Trends enrichment ───────────────────────────────────────────
    trends_keyword: str = "кроссовки"
    trends_geo: str = "RU"
    trends_tz: str = "-180"
    trends_max_chunk_days: int = 250
    trends_overlap_days: int = 80
    trends_date_start: str = "2024-07-04"
    trends_date_end: str = "2025-07-21"


# ── Singleton instance ─────────────────────────────────────────────────────
CFG = Config()


# ── Metrics (must be in importable module for Ray pickling) ───────────────
import numpy as np


def wmae(y_true, y_pred):
    """Weighted MAE: w=7 for positive days, w=1 for zero days."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    w = np.where(y_true > 0, CFG.wmae_weight_positive, CFG.wmae_weight_zero)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)


def f2_score(y_true, y_pred) -> float:
    """F-beta (beta=2) for positive class = 1."""
    from sklearn.metrics import fbeta_score
    return float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0))


def get_f2_scorer():
    """Create AutoGluon scorer for F2. Call this instead of using a global make_scorer."""
    from autogluon.core.metrics import make_scorer
    return make_scorer(
        name="f2_score",
        score_func=f2_score,
        optimum=1,
        greater_is_better=True,
        needs_class=True,
    )
