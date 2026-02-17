"""
WB Hackathon — Tabular Hurdle Model Pipeline
=============================================
Step 1: Binary classifier — P(qty > 0)
Step 2: Regressor on positive-only data — predicts qty/3 (k)
Step 3: Threshold optimization on validation set using wMAE
Step 4: Generate submission
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

import torch
from autogluon.tabular import TabularPredictor
from features_365 import build_features, FEATURE_COLS
from config import CFG, set_global_seed, wmae, f2_score, get_f2_scorer

set_global_seed(CFG.seed)


# ============================================================
# Config (from centralized config.py)
# ============================================================
VAL_DAYS = CFG.val_days
TUNING_DAYS = CFG.tuning_days
CLASSIFIER_TIME_LIMIT = CFG.classifier_time_limit
REGRESSOR_TIME_LIMIT = CFG.regressor_time_limit
CLASSIFIER_PRESET = CFG.classifier_preset
REGRESSOR_PRESET = CFG.regressor_preset
OUTPUT_DIR = CFG.tabular_model_dir
# Clean previous models to avoid "Learner is already fit" errors on rerun
import shutil
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"[init] Removed existing {OUTPUT_DIR}/")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Metric (imported from config.py for Ray pickling compatibility)
# ============================================================
f2 = get_f2_scorer()


# ============================================================
# 1. Load data
# ============================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_parquet("train.parquet")
test = pd.read_parquet("test.parquet")
train["dt"] = pd.to_datetime(train["dt"])
test["dt"] = pd.to_datetime(test["dt"])

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. Time-based 3-way split: train / tuning / holdout
#    - tuning  → passed to fit() for early stopping & model selection
#    - holdout → used for threshold optimisation & final wMAE eval
# ============================================================
print("\n" + "=" * 70)
print("CREATING 3-WAY TIME SPLIT  (train → tuning → holdout)")
print("=" * 70)

holdout_start = train["dt"].max() - pd.Timedelta(days=VAL_DAYS - 1)
tuning_start = holdout_start - pd.Timedelta(days=TUNING_DAYS)

train_split = train[train["dt"] < tuning_start].copy()
tuning_split = train[(train["dt"] >= tuning_start) & (train["dt"] < holdout_start)].copy()
val_split = train[train["dt"] >= holdout_start].copy()

print(f"Train split:   {train_split.shape} ({train_split['dt'].min()} to {train_split['dt'].max()})")
print(f"Tuning split:  {tuning_split.shape} ({tuning_split['dt'].min()} to {tuning_split['dt'].max()})")
print(f"Holdout split: {val_split.shape} ({val_split['dt'].min()} to {val_split['dt'].max()})")
print(f"Holdout items: {val_split['nm_id'].nunique()}")

# ============================================================
# 3. Feature engineering
# ============================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

print("Building train_split features...")
train_feat = build_features(train_split, train_split, is_train=True)

print("Building tuning_split features...")
tuning_feat = build_features(tuning_split, train_split, is_train=False)

print("Building holdout_split features...")
val_feat = build_features(val_split, train_split, is_train=False)

print("Building test features (using full train)...")
test_feat = build_features(test, train, is_train=False)

# Filter to available feature columns
available_features = [c for c in FEATURE_COLS if c in train_feat.columns]
missing = [c for c in FEATURE_COLS if c not in train_feat.columns]
if missing:
    print(f"Warning: missing features: {missing}")
print(f"Using {len(available_features)} features")

# ============================================================
# 3b. Sample weighting — give more weight to recent observations
# ============================================================
if CFG.use_sample_weight:
    print("\n  Computing sample weights (exponential decay)...")
    half_life = CFG.sample_weight_half_life
    max_date = train_feat["dt"].max()
    days_ago = (max_date - train_feat["dt"]).dt.days
    train_feat["sample_weight"] = np.exp(-np.log(2) * days_ago / half_life)
    print(f"  Half-life: {half_life} days")
    print(f"  Weight range: {train_feat['sample_weight'].min():.4f} — {train_feat['sample_weight'].max():.4f}")
    print(f"  Median weight: {train_feat['sample_weight'].median():.4f}")
    print(f"  Effective sample size: {train_feat['sample_weight'].sum():.0f} "
          f"(of {len(train_feat)} actual rows)")

# ============================================================
# 4. Train binary classifier
# ============================================================
print("\n" + "=" * 70)
print("TRAINING BINARY CLASSIFIER")
print("=" * 70)

# Prepare classifier data
train_feat["is_sale"] = (train_feat["qty"] > 0).astype(int)
tuning_feat["is_sale"] = (tuning_feat["qty"] > 0).astype(int)
val_feat["is_sale"] = (val_feat["qty"] > 0).astype(int)

# Class-based sample weight: w=7 for sales, w=1 for non-sales (mirrors wMAE asymmetry)
train_feat["clf_weight"] = np.where(train_feat["is_sale"] == 1, 7.0, 1.0)
tuning_feat["clf_weight"] = np.where(tuning_feat["is_sale"] == 1, 7.0, 1.0)

clf_cols = available_features + ["is_sale", "clf_weight"]
clf_train = train_feat[clf_cols].copy()
clf_tuning = tuning_feat[clf_cols].copy()
clf_val = val_feat[available_features + ["is_sale"]].copy()

print(f"Classifier train:  {clf_train.shape}, positive rate: {clf_train['is_sale'].mean():.4f}")
print(f"Classifier tuning: {clf_tuning.shape}, positive rate: {clf_tuning['is_sale'].mean():.4f}")
print(f"Classifier holdout:{clf_val.shape}, positive rate: {clf_val['is_sale'].mean():.4f}")
print(f"  Using weighted log_loss: w=7 for sales, w=1 for non-sales")

clf_predictor_kwargs = dict(
    label="is_sale",
    problem_type="binary",
    eval_metric="log_loss",
    sample_weight="clf_weight",
    path=os.path.join(OUTPUT_DIR, "classifier"),
    verbosity=2,
)

clf_predictor = TabularPredictor(**clf_predictor_kwargs)

clf_predictor.fit(
    train_data=clf_train,
    tuning_data=clf_tuning,
    use_bag_holdout=True,
    time_limit=CLASSIFIER_TIME_LIMIT,
    presets=CLASSIFIER_PRESET,
    ag_args_fit={"random_seed": CFG.seed,
    "num_gpus": torch.cuda.device_count()},
)

# Predict probabilities on holdout (unseen by fit)
val_proba = clf_predictor.predict_proba(clf_val[available_features])
val_p_sale = val_proba[1].values  # P(sale)

print(f"\nClassifier leaderboard (on holdout):")
print(clf_predictor.leaderboard(clf_val, silent=True).to_string())

# ============================================================
# 5. Train regressor on positive-only data
# ============================================================
print("\n" + "=" * 70)
print("TRAINING REGRESSOR (positive qty only)")
print("=" * 70)

# Target: k = qty / 3 (since qty is always multiple of 3)
train_pos = train_feat[train_feat["qty"] > 0].copy()
train_pos["k"] = train_pos["qty"] / 3

tuning_pos = tuning_feat[tuning_feat["qty"] > 0].copy()
tuning_pos["k"] = tuning_pos["qty"] / 3

val_pos = val_feat[val_feat["qty"] > 0].copy()
val_pos["k"] = val_pos["qty"] / 3

reg_cols = available_features + ["k"]
if CFG.use_sample_weight:
    reg_cols = reg_cols + ["sample_weight"]

reg_train = train_pos[reg_cols].copy()
reg_tuning = tuning_pos[available_features + ["k"]].copy()
reg_val = val_pos[available_features + ["k"]].copy()

print(f"Regressor train:  {reg_train.shape}")
print(f"Regressor tuning: {reg_tuning.shape}")
print(f"Regressor holdout:{reg_val.shape}")
print(f"k distribution: mean={reg_train['k'].mean():.2f}, median={reg_train['k'].median():.2f}, max={reg_train['k'].max():.0f}")

reg_predictor_kwargs = dict(
    label="k",
    problem_type="regression",
    eval_metric="mean_absolute_error",
    path=os.path.join(OUTPUT_DIR, "regressor"),
    verbosity=2,
)
if CFG.use_sample_weight:
    reg_predictor_kwargs["sample_weight"] = "sample_weight"
    print("  Using sample_weight for regressor")

reg_predictor = TabularPredictor(**reg_predictor_kwargs)

reg_predictor.fit(
    train_data=reg_train,
    tuning_data=reg_tuning,
    use_bag_holdout=True,
    time_limit=REGRESSOR_TIME_LIMIT,
    presets=REGRESSOR_PRESET,
    ag_args_fit={"random_seed": CFG.seed,
    "num_gpus": torch.cuda.device_count()},
)

print(f"\nRegressor leaderboard (on holdout):")
print(reg_predictor.leaderboard(reg_val, silent=True).to_string())

# ============================================================
# 6. Threshold optimization on validation
# ============================================================
print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

# For all val records, predict qty
# Regressor predicts k for ALL val records (not just positive), then we scale
val_k_pred = reg_predictor.predict(val_feat[available_features])
val_qty_pred_raw = val_k_pred.values * 3  # Convert k back to qty

y_true_val = val_feat["qty"].values
thresholds = np.arange(CFG.threshold_min, CFG.threshold_max, CFG.threshold_step)

# ------------------------------------------------------------------
# 6a. Global threshold (always computed as baseline for comparison)
# ------------------------------------------------------------------
best_threshold = 0.5
best_wmae_global = float("inf")
results = []

for thr in thresholds:
    y_pred = np.where(val_p_sale >= thr, val_qty_pred_raw, 0)
    y_pred = np.round(y_pred / 3) * 3
    y_pred = np.maximum(y_pred, 0)
    score = wmae(y_true_val, y_pred)
    results.append({"threshold": thr, "wmae": score})
    if score < best_wmae_global:
        best_wmae_global = score
        best_threshold = thr

print(f"Global best threshold: {best_threshold:.2f}")
print(f"Global wMAE on validation: {best_wmae_global:.6f}")

# Compare with baselines
baseline_zeros = wmae(y_true_val, np.zeros_like(y_true_val))
baseline_mean = wmae(y_true_val, np.full_like(y_true_val, y_true_val.mean(), dtype=float))
print(f"\nBaseline 'all zeros': {baseline_zeros:.6f}")
print(f"Baseline 'global mean': {baseline_mean:.6f}")
print(f"Improvement over zeros (global thr): {(baseline_zeros - best_wmae_global) / baseline_zeros * 100:.1f}%")

y_pred_reg_only = np.round(val_qty_pred_raw / 3) * 3
y_pred_reg_only = np.maximum(y_pred_reg_only, 0)
wmae_reg_only = wmae(y_true_val, y_pred_reg_only)
print(f"Regressor only (no threshold): wMAE = {wmae_reg_only:.6f}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_results.csv"), index=False)

# ------------------------------------------------------------------
# 6b. Grouped threshold by item_pct_nonzero buckets
# ------------------------------------------------------------------
group_thresholds = {}  # group_label -> best_threshold

if CFG.use_grouped_threshold:
    print("\n" + "-" * 50)
    print("GROUPED THRESHOLD OPTIMIZATION (by item_pct_nonzero)")
    print("-" * 50)

    bins = CFG.grouped_threshold_bins
    bin_labels = CFG.grouped_threshold_labels

    val_feat["sale_group"] = pd.cut(
        val_feat["item_pct_nonzero"].fillna(0),
        bins=bins, labels=bin_labels, include_lowest=True,
    )

    for group in bin_labels:
        mask = (val_feat["sale_group"] == group).values
        n_group = mask.sum()
        if n_group == 0:
            group_thresholds[group] = best_threshold  # fallback to global
            print(f"  {group:15s}: n=0, using global threshold {best_threshold:.2f}")
            continue

        y_true_g = y_true_val[mask]
        p_sale_g = val_p_sale[mask]
        qty_pred_g = val_qty_pred_raw[mask]

        best_thr_g, best_wmae_g = best_threshold, float("inf")
        for thr in thresholds:
            y_pred_g = np.where(p_sale_g >= thr, qty_pred_g, 0)
            y_pred_g = np.round(y_pred_g / 3) * 3
            y_pred_g = np.maximum(y_pred_g, 0)
            score = wmae(y_true_g, y_pred_g)
            if score < best_wmae_g:
                best_wmae_g = score
                best_thr_g = thr

        group_thresholds[group] = round(float(best_thr_g), 2)
        pct_nz = y_true_g[y_true_g > 0].shape[0] / max(len(y_true_g), 1) * 100
        print(f"  {group:15s}: threshold={best_thr_g:.2f}, wMAE={best_wmae_g:.4f}, "
              f"n={n_group}, val_pct_positive={pct_nz:.1f}%")

    # Compute grouped wMAE for comparison with global
    y_pred_grouped = np.zeros_like(y_true_val, dtype=float)
    for group, thr in group_thresholds.items():
        mask = (val_feat["sale_group"] == group).values
        y_pred_grouped[mask] = np.where(val_p_sale[mask] >= thr, val_qty_pred_raw[mask], 0)
    y_pred_grouped = np.round(y_pred_grouped / 3) * 3
    y_pred_grouped = np.maximum(y_pred_grouped, 0)
    best_wmae_grouped = wmae(y_true_val, y_pred_grouped)

    print(f"\n  Grouped wMAE: {best_wmae_grouped:.6f}")
    print(f"  Global  wMAE: {best_wmae_global:.6f}")
    delta = best_wmae_global - best_wmae_grouped
    print(f"  Improvement from grouping: {delta:.6f} "
          f"({'BETTER' if delta > 0 else 'WORSE — falling back to global'})")

    if delta <= 0:
        print("  Grouped threshold did not improve — using global threshold for all groups.")
        CFG.use_grouped_threshold = False
        group_thresholds = {}

    best_wmae = best_wmae_grouped if CFG.use_grouped_threshold else best_wmae_global
else:
    best_wmae = best_wmae_global

# ============================================================
# 7. Retrain on FULL training data (using thresholds from holdout)
# ============================================================
print("\n" + "=" * 70)
print("RETRAINING ON FULL TRAIN DATA")
print("=" * 70)

print("Building features on full training data...")
full_train_feat = build_features(train, train, is_train=True)

# --- Retrain classifier ---
print("\nRetraining classifier on full train...")
full_train_feat["is_sale"] = (full_train_feat["qty"] > 0).astype(int)
full_train_feat["clf_weight"] = np.where(full_train_feat["is_sale"] == 1, 7.0, 1.0)

full_clf_cols = available_features + ["is_sale", "clf_weight"]
full_clf_train = full_train_feat[full_clf_cols].copy()

print(f"  Full classifier train: {full_clf_train.shape}, positive rate: {full_clf_train['is_sale'].mean():.4f}")

# Clean old classifier dir and create new one
full_clf_path = os.path.join(OUTPUT_DIR, "classifier_final")
if os.path.exists(full_clf_path):
    shutil.rmtree(full_clf_path)

full_clf_predictor = TabularPredictor(
    label="is_sale",
    problem_type="binary",
    eval_metric="log_loss",
    sample_weight="clf_weight",
    path=full_clf_path,
    verbosity=2,
)

full_clf_predictor.fit(
    train_data=full_clf_train,
    time_limit=CLASSIFIER_TIME_LIMIT,
    presets=CLASSIFIER_PRESET,
    ag_args_fit={"random_seed": CFG.seed,
    "num_gpus": torch.cuda.device_count()},
)

# --- Retrain regressor ---
print("\nRetraining regressor on full train (positive only)...")
full_train_pos = full_train_feat[full_train_feat["qty"] > 0].copy()
full_train_pos["k"] = full_train_pos["qty"] / 3

full_reg_cols = available_features + ["k"]
full_reg_train = full_train_pos[full_reg_cols].copy()

print(f"  Full regressor train: {full_reg_train.shape}")
print(f"  k distribution: mean={full_reg_train['k'].mean():.2f}, median={full_reg_train['k'].median():.2f}, max={full_reg_train['k'].max():.0f}")

full_reg_path = os.path.join(OUTPUT_DIR, "regressor_final")
if os.path.exists(full_reg_path):
    shutil.rmtree(full_reg_path)

full_reg_predictor = TabularPredictor(
    label="k",
    problem_type="regression",
    eval_metric="mean_absolute_error",
    path=full_reg_path,
    verbosity=2,
)

full_reg_predictor.fit(
    train_data=full_reg_train,
    time_limit=REGRESSOR_TIME_LIMIT,
    presets=REGRESSOR_PRESET,
    ag_args_fit={"random_seed": CFG.seed,
    "num_gpus": torch.cuda.device_count()},
)

print("\nFull-data retraining complete.")
print(f"  Classifier: {full_clf_train.shape[0]} rows (was {clf_train.shape[0]})")
print(f"  Regressor:  {full_reg_train.shape[0]} rows (was {reg_train.shape[0]})")

# ============================================================
# 8. Generate test predictions + submission (using retrained models)
# ============================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSION (with full-data models)")
print("=" * 70)

# Classifier predictions on test (using retrained model)
test_proba = full_clf_predictor.predict_proba(test_feat[available_features])
test_p_sale = test_proba[1].values

# Regressor predictions on test (using retrained model)
test_k_pred = full_reg_predictor.predict(test_feat[available_features])
test_qty_pred_raw = test_k_pred.values * 3

# Apply threshold(s)
if CFG.use_grouped_threshold and group_thresholds:
    print("Applying grouped thresholds by item_pct_nonzero...")
    bins = CFG.grouped_threshold_bins
    bin_labels = CFG.grouped_threshold_labels

    test_feat["sale_group"] = pd.cut(
        test_feat["item_pct_nonzero"].fillna(0),
        bins=bins, labels=bin_labels, include_lowest=True,
    )

    test_qty_pred = np.zeros(len(test_feat), dtype=float)
    for group, thr in group_thresholds.items():
        mask = (test_feat["sale_group"] == group).values
        test_qty_pred[mask] = np.where(
            test_p_sale[mask] >= thr, test_qty_pred_raw[mask], 0
        )
        n_group = mask.sum()
        n_nz = (test_qty_pred[mask] > 0).sum()
        print(f"  {group:15s}: thr={thr:.2f}, n={n_group}, predicted_nonzero={n_nz}")
else:
    print(f"Applying global threshold: {best_threshold:.2f}")
    test_qty_pred = np.where(test_p_sale >= best_threshold, test_qty_pred_raw, 0)

test_qty_pred = np.round(test_qty_pred / 3) * 3
test_qty_pred = np.maximum(test_qty_pred, 0).astype(int)

# Build submission
submission = pd.DataFrame({
    "nm_id": test_feat["nm_id"],
    "dt": test_feat["dt"].dt.strftime("%Y-%m-%d"),
    "qty": test_qty_pred,
})

# Verify structure
sample_sub = pd.read_csv("sample_submission.csv")
assert len(submission) == len(sample_sub), f"Length mismatch: {len(submission)} vs {len(sample_sub)}"

submission.to_csv("submission_tabular.csv", index=False)
print(f"\nSubmission saved: submission_tabular.csv")
print(f"Shape: {submission.shape}")
print(f"Qty distribution: {submission['qty'].describe()}")
print(f"% zero: {(submission['qty'] == 0).mean() * 100:.1f}%")
print(f"Non-zero values: {submission[submission['qty'] > 0]['qty'].value_counts().head(10)}")

# Save config
config = {
    "best_global_threshold": float(best_threshold),
    "use_grouped_threshold": bool(CFG.use_grouped_threshold),
    "group_thresholds": {k: float(v) for k, v in group_thresholds.items()} if group_thresholds else {},
    "best_wmae_val": float(best_wmae),
    "best_wmae_global": float(best_wmae_global),
    "baseline_zeros": float(baseline_zeros),
    "wmae_reg_only": float(wmae_reg_only),
    "val_days": VAL_DAYS,
    "tuning_days": TUNING_DAYS,
    "n_features": len(available_features),
    "classifier_preset": CLASSIFIER_PRESET,
    "regressor_preset": REGRESSOR_PRESET,
    "use_sample_weight": CFG.use_sample_weight,
    "sample_weight_half_life": CFG.sample_weight_half_life if CFG.use_sample_weight else None,
    "seed": CFG.seed,
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Summary
print("\n" + "=" * 70)
print("DONE!")
print(f"Models saved to: {OUTPUT_DIR}/")
print(f"  Threshold-optimized models: classifier/ + regressor/")
print(f"  Final full-data models:     classifier_final/ + regressor_final/")
print(f"Submission: submission_tabular.csv")
if CFG.use_grouped_threshold and group_thresholds:
    print(f"Thresholds (grouped): {group_thresholds}")
else:
    print(f"Threshold (global): {best_threshold:.2f}")
print(f"Best wMAE (holdout, split models): {best_wmae:.6f}")
if CFG.use_sample_weight:
    print(f"Sample weighting: half_life={CFG.sample_weight_half_life} days")
print("=" * 70)