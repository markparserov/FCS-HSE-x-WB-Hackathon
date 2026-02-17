"""
Replicate tabular submission using already trained models.

Requirements:
- models trained by train_tabular.py are present in CFG.tabular_model_dir:
    - classifier_final/
    - regressor_final/
    - config.json  (with thresholds and metrics)
- train.parquet, test.parquet, sample_submission.csv are in current directory
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from config import CFG, set_global_seed
from features_365 import build_features, FEATURE_COLS

warnings.filterwarnings("ignore")

def main():
    # ── 1. Reproducibility ───────────────────────────────────────────────
    set_global_seed(CFG.seed)

    model_dir = CFG.tabular_model_dir
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found at {config_path}. "
            f"Make sure you have already run train_tabular.py."
        )

    with open(config_path, "r") as f:
        cfg_saved = json.load(f)

    best_global_threshold = cfg_saved.get("best_global_threshold", 0.5)
    use_grouped_threshold = bool(cfg_saved.get("use_grouped_threshold", False))
    group_thresholds = cfg_saved.get("group_thresholds", {}) or {}

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # You can also use CFG.train_path / CFG.test_path if желаешь
    train = pd.read_parquet(CFG.train_path)
    test = pd.read_parquet(CFG.test_path)

    train["dt"] = pd.to_datetime(train["dt"])
    test["dt"] = pd.to_datetime(test["dt"])

    print(f"Train: {train.shape}, Test: {test.shape}")

    # ── 2. Feature engineering (как в train_tabular, но без обучения) ─────
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING FOR TEST")
    print("=" * 70)

    # Для теста нужны исторические данные train для лагов/статик и т.д.
    print("Building test features (using full train)...")
    test_feat = build_features(test, train, is_train=False)

    # Выбираем те же фичи, что и при обучении (могут отсутствовать из-за NaN/типов)
    available_features = [c for c in FEATURE_COLS if c in test_feat.columns]
    missing = [c for c in FEATURE_COLS if c not in test_feat.columns]
    if missing:
        print(f"Warning: missing features in test set (ignored): {missing}")
    print(f"Using {len(available_features)} features")

    # ── 3. Load trained models ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LOADING TRAINED MODELS")
    print("=" * 70)

    clf_path = os.path.join(model_dir, "classifier_final")
    reg_path = os.path.join(model_dir, "regressor_final")

    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Classifier model directory not found: {clf_path}")
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Regressor model directory not found: {reg_path}")

    print(f"Loading classifier from: {clf_path}")
    full_clf_predictor = TabularPredictor.load(clf_path)

    print(f"Loading regressor from: {reg_path}")
    full_reg_predictor = TabularPredictor.load(reg_path)

    # ── 4. Predictions on test ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    # print(full_clf_predictor.leaderboard())
    # print(full_reg_predictor.leaderboard())
    
    # Model selection for experiments
    clf_model = "LightGBMLarge"  # Classifier model
    reg_model = "WeightedEnsemble_L2"  # Regressor model
    # CatBoost, ExtraTreesEntr, ExtraTreesGini, LightGBM, LightGBMLarge, LightGBMXT, NeuralNetTorch, RandomForestEntr, RandomForestGini, XGBoost, WeightedEnsemble_L2
    # CatBoost, ExtraTreesMSE, LightGBM, LightGBMLarge, LightGBMXT, NeuralNetTorch, RandomForestMSE, XGBoost, WeightedEnsemble_L2

    # Classifier: P(sale)
    test_proba = full_clf_predictor.predict_proba(test_feat[available_features], model=clf_model)
    # Column 1 = positive class
    test_p_sale = test_proba[1].values

    # Regressor: k = qty / 3  → qty_raw = k * 3
    test_k_pred = full_reg_predictor.predict(test_feat[available_features], model=reg_model)
    test_qty_pred_raw = test_k_pred.values * 3

    # ── 5. Apply threshold(s) (global or grouped), как в train_tabular.py ─
    if use_grouped_threshold and len(group_thresholds) > 0:
        print("Applying grouped thresholds by item_pct_nonzero...")

        bins = CFG.grouped_threshold_bins
        bin_labels = CFG.grouped_threshold_labels

        test_feat["sale_group"] = pd.cut(
            test_feat["item_pct_nonzero"].fillna(0),
            bins=bins,
            labels=bin_labels,
            include_lowest=True,
        )

        test_qty_pred = np.zeros(len(test_feat), dtype=float)

        for group in bin_labels:
            thr = group_thresholds.get(group)
            if thr is None:
                # safety fallback: global threshold
                thr = best_global_threshold

            mask = (test_feat["sale_group"] == group).values
            if mask.sum() == 0:
                continue

            test_qty_pred[mask] = np.where(
                test_p_sale[mask] >= thr,
                test_qty_pred_raw[mask],
                0,
            )
            n_group = int(mask.sum())
            n_nz = int((test_qty_pred[mask] > 0).sum())
            print(f"  {group:15s}: thr={thr:.2f}, n={n_group}, predicted_nonzero={n_nz}")

    else:
        print(f"Applying global threshold: {best_global_threshold:.2f}")
        test_qty_pred = np.where(test_p_sale >= best_global_threshold,
                                 test_qty_pred_raw, 0)

    # Кратность 3 и неотрицательные целые
    test_qty_pred = np.round(test_qty_pred / 3) * 3
    test_qty_pred = np.maximum(test_qty_pred, 0).astype(int)

    # ── 6. Build submission ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BUILDING SUBMISSION")
    print("=" * 70)

    submission = pd.DataFrame(
        {
            "nm_id": test_feat["nm_id"],
            "dt": test_feat["dt"].dt.strftime("%Y-%m-%d"),
            "qty": test_qty_pred,
        }
    )

    sample_sub = pd.read_csv(CFG.sample_submission_path)
    assert len(submission) == len(sample_sub), (
        f"Length mismatch: {len(submission)} vs {len(sample_sub)}"
    )

    # Insert model names before file extension
    base, ext = os.path.splitext(CFG.submission_tabular)
    model_str = f"{clf_model}_{reg_model}"
    out_path = f"{base}_{model_str}{ext}"
    submission.to_csv(out_path, index=False)

    print(f"\nSubmission saved: {out_path}")
    print(f"Shape: {submission.shape}")
    print(f"Qty distribution:\n{submission['qty'].describe()}")
    print(f"% zero: {(submission['qty'] == 0).mean() * 100:.1f}%")
    print(
        "Non-zero values (top 10):\n",
        submission[submission["qty"] > 0]["qty"].value_counts().head(10),
    )


if __name__ == "__main__":
    main()