"""
Feature engineering for WB Hackathon — Sneakers Sales Prediction.
Shared module used by both TabularPredictor and TimeSeriesPredictor pipelines.
"""

import pandas as pd
import numpy as np
import holidays
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Existing feature functions (calendar, item static, price, leftovers)
# ============================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and RU holiday features."""
    df = df.copy()
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["day_of_month"] = df["dt"].dt.day
    df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
    df["month"] = df["dt"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Russian holidays
    ru_holidays = holidays.Russia(years=[2024, 2025])
    holiday_dates = set(ru_holidays.keys())
    df["is_holiday"] = df["dt"].dt.date.apply(lambda d: d in holiday_dates).astype(int)

    # Days to/from nearest holiday
    all_hol = sorted(holiday_dates)
    all_hol_ts = pd.to_datetime(all_hol)

    def days_to_nearest_holiday(date):
        diffs = np.abs((all_hol_ts - date).days)
        return int(diffs.min())

    unique_dates = df["dt"].dt.normalize().unique()
    date_to_hol_dist = {d: days_to_nearest_holiday(d) for d in unique_dates}
    df["days_to_holiday"] = df["dt"].dt.normalize().map(date_to_hol_dist)

    return df


def add_item_static_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-item static features computed from train history."""
    item_stats = train_df.groupby("nm_id").agg(
        item_total_qty=("qty", "sum"),
        item_mean_qty=("qty", "mean"),
        item_std_qty=("qty", "std"),
        item_max_qty=("qty", "max"),
        item_pct_nonzero=("qty", lambda x: (x > 0).mean()),
        item_mean_price=("price", "mean"),
        item_std_price=("price", "std"),
        item_min_price=("price", "min"),
        item_max_price=("price", "max"),
        item_mean_leftovers=("prev_leftovers", "mean"),
        item_n_days=("dt", "nunique"),
        item_pct_promo=("is_promo", "mean"),
        item_first_date=("dt", "min"),
        item_last_date=("dt", "max"),
    ).reset_index()

    item_stats["item_std_qty"] = item_stats["item_std_qty"].fillna(0)
    item_stats["item_std_price"] = item_stats["item_std_price"].fillna(0)
    item_stats["item_price_range_pct"] = (
        (item_stats["item_max_price"] - item_stats["item_min_price"])
        / item_stats["item_mean_price"].clip(lower=1)
        * 100
    )
    item_stats["item_cv_qty"] = (
        item_stats["item_std_qty"] / item_stats["item_mean_qty"].clip(lower=0.001)
    )

    df = df.merge(item_stats, on="nm_id", how="left")
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-related features relative to item history."""
    df = df.copy()
    # Price vs item average
    df["price_vs_item_mean"] = df["price"] / df["item_mean_price"].clip(lower=1)
    df["price_vs_item_min"] = df["price"] / df["item_min_price"].clip(lower=1)
    df["price_vs_item_max"] = df["price"] / df["item_max_price"].clip(lower=1)
    # Price discount proxy
    df["price_discount_pct"] = (1 - df["price_vs_item_max"]) * 100
    return df


def add_leftovers_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prev_leftovers derived features."""
    df = df.copy()
    df["log_leftovers"] = np.log1p(df["prev_leftovers"])
    df["leftovers_vs_item_mean"] = (
        df["prev_leftovers"] / df["item_mean_leftovers"].clip(lower=1)
    )
    return df


# ============================================================
# NEW feature functions
# ============================================================

def add_forward_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward delta of prev_leftovers — the strongest predictive signal.

    fwd_delta_lo ≈ qty in ~82% of cases (exact when no restocking).
    Available for all days except the last day of each item in the dataset.
    Correlation with qty on validation: ~0.52 (vs 0.37 for prev_leftovers).
    """
    df = df.copy()
    df = df.sort_values(["nm_id", "dt"])
    next_lo = df.groupby("nm_id")["prev_leftovers"].shift(-1)
    # Clipped version: lower bound on sales (always >= 0)
    df["fwd_delta_lo"] = (df["prev_leftovers"] - next_lo).clip(lower=0)
    # Raw version: negative means restocking happened (model learns to correct)
    df["fwd_delta_lo_raw"] = df["prev_leftovers"] - next_lo
    return df


def add_price_change_features(df: pd.DataFrame, train_df: pd.DataFrame,
                               is_train: bool = True) -> pd.DataFrame:
    """
    Add backward delta of leftovers and day-over-day price changes.
    For non-train data, bridges from the last training days for proper computation.
    """
    df = df.copy()

    if not is_train:
        # Bridge: prepend last 7 days of train per item for shift computation
        train_sorted = train_df.sort_values(["nm_id", "dt"])
        target_items = set(df["nm_id"].unique())
        train_tail = (
            train_sorted[train_sorted["nm_id"].isin(target_items)]
            .groupby("nm_id").tail(7)
        )

        bridge_cols = ["nm_id", "dt", "price", "prev_leftovers"]
        bridge = pd.concat([
            train_tail[bridge_cols],
            df[bridge_cols]
        ]).sort_values(["nm_id", "dt"]).reset_index(drop=True)

        bridge["_prev_lo"] = bridge.groupby("nm_id")["prev_leftovers"].shift(1)
        bridge["_prev_price"] = bridge.groupby("nm_id")["price"].shift(1)
        bridge["_price_7d"] = bridge.groupby("nm_id")["price"].shift(7)

        # Filter back to target rows only
        target_min_dt = df["dt"].min()
        bridge_target = bridge[bridge["dt"] >= target_min_dt][
            ["nm_id", "dt", "_prev_lo", "_prev_price", "_price_7d"]
        ]
        df = df.merge(bridge_target, on=["nm_id", "dt"], how="left")
    else:
        df = df.sort_values(["nm_id", "dt"])
        df["_prev_lo"] = df.groupby("nm_id")["prev_leftovers"].shift(1)
        df["_prev_price"] = df.groupby("nm_id")["price"].shift(1)
        df["_price_7d"] = df.groupby("nm_id")["price"].shift(7)

    # Backward delta (positive = stock decreased from previous day = likely sales)
    df["bwd_delta_lo"] = df["_prev_lo"] - df["prev_leftovers"]

    # Price changes
    df["price_change_1d"] = (
        (df["price"] - df["_prev_price"]) / df["_prev_price"].clip(lower=1)
    )
    df["price_change_7d"] = (
        (df["price"] - df["_price_7d"]) / df["_price_7d"].clip(lower=1)
    )
    df["price_dropped"] = (df["price_change_1d"] < -0.05).astype(int)
    df["price_dropped_big"] = (df["price_change_1d"] < -0.15).astype(int)

    # Handle NaN in binary features (first day of each item has no previous)
    df["price_dropped"] = df["price_dropped"].fillna(0).astype(int)
    df["price_dropped_big"] = df["price_dropped_big"].fillna(0).astype(int)

    # Cleanup temp columns
    df = df.drop(columns=["_prev_lo", "_prev_price", "_price_7d"], errors="ignore")
    return df


def add_sell_rate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Add expected qty based on historical item sell-through rate."""
    df = df.copy()

    item_rates = train_df.groupby("nm_id").agg(
        _total_qty=("qty", "sum"),
        _total_leftovers=("prev_leftovers", "sum"),
    )
    item_rates["item_sell_rate"] = (
        item_rates["_total_qty"] / item_rates["_total_leftovers"].clip(lower=1)
    )
    item_rates = item_rates[["item_sell_rate"]].reset_index()

    df = df.merge(item_rates, on="nm_id", how="left")
    df["item_sell_rate"] = df["item_sell_rate"].fillna(0)
    df["expected_qty"] = df["prev_leftovers"] * df["item_sell_rate"]

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between key predictors."""
    df = df.copy()

    # Promo + price drop (strongest combined effect: 21.9% nonzero rate)
    if "price_dropped" in df.columns:
        df["promo_and_drop"] = df["is_promo"] * df["price_dropped"]

    # Promo * discount percentage
    if "price_discount_pct" in df.columns:
        df["promo_x_discount"] = df["is_promo"] * df["price_discount_pct"]

    # Leftovers * promo
    df["leftovers_x_promo"] = df["prev_leftovers"] * df["is_promo"]

    # Leftovers * historical nonzero rate (expected demand proxy)
    if "item_pct_nonzero" in df.columns:
        df["leftovers_x_pct_nz"] = df["prev_leftovers"] * df["item_pct_nonzero"]

    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-level daily aggregate features."""
    df = df.copy()

    daily = df.groupby("dt").agg(
        market_pct_promo=("is_promo", "mean"),
        market_mean_price=("price", "mean"),
        market_mean_leftovers=("prev_leftovers", "mean"),
    ).reset_index()

    df = df.merge(daily, on="dt", how="left")
    df["price_vs_market"] = df["price"] / df["market_mean_price"].clip(lower=1)

    return df


# ============================================================
# Existing lag and promo features (updated to use is_train)
# ============================================================

def add_lag_features(df: pd.DataFrame, train_df: pd.DataFrame,
                     is_train: bool = True) -> pd.DataFrame:
    """
    Add lag features computed from training data.
    For each (nm_id, dt), compute rolling stats from the item's history
    strictly before that date.

    Uses is_train flag (not 'qty' in df.columns) to properly handle
    tuning/val splits which have qty but need the non-train merge path.
    """
    df = df.copy()

    # Build per-item time series from train
    train_sorted = train_df.sort_values(["nm_id", "dt"])

    # Preserve original index for safe alignment after rolling
    train_sorted = train_sorted.reset_index(drop=True)

    # Compute rolling aggregates on train per item
    rolling_features = []
    for window in [7, 14, 30]:
        rolled = (
            train_sorted.groupby("nm_id")["qty"]
            .rolling(window, min_periods=1)
            .agg(["mean", "sum", "max", "std"])
        )
        # rolled has a MultiIndex (nm_id, original_idx) — drop nm_id level
        rolled = rolled.droplevel(0)
        rolled.columns = [f"qty_mean_{window}d", f"qty_sum_{window}d",
                          f"qty_max_{window}d", f"qty_std_{window}d"]
        rolled[f"qty_std_{window}d"] = rolled[f"qty_std_{window}d"].fillna(0)
        # Align back by original index — guaranteed correct order
        rolled["nm_id"] = train_sorted["nm_id"].values
        rolled["dt"] = train_sorted["dt"].values
        rolling_features.append(rolled.reset_index(drop=True))

    # Rolling pct_nonzero
    for window in [7, 14, 30]:
        rolled_nz = (
            train_sorted.groupby("nm_id")["qty"]
            .rolling(window, min_periods=1)
            .apply(lambda x: (x > 0).mean(), raw=True)
        )
        rolled_nz = rolled_nz.droplevel(0)
        rolled_nz = rolled_nz.to_frame(name=f"pct_nonzero_{window}d")
        rolled_nz["nm_id"] = train_sorted["nm_id"].values
        rolled_nz["dt"] = train_sorted["dt"].values
        rolling_features.append(rolled_nz.reset_index(drop=True))

    # Rolling price features
    for window in [7, 14]:
        rolled_p = (
            train_sorted.groupby("nm_id")["price"]
            .rolling(window, min_periods=1)
            .mean()
        )
        rolled_p = rolled_p.droplevel(0)
        rolled_p = rolled_p.to_frame(name=f"price_mean_{window}d")
        rolled_p["nm_id"] = train_sorted["nm_id"].values
        rolled_p["dt"] = train_sorted["dt"].values
        rolling_features.append(rolled_p.reset_index(drop=True))

    # Rolling leftovers
    rolled_lo = (
        train_sorted.groupby("nm_id")["prev_leftovers"]
        .rolling(7, min_periods=1)
        .mean()
    )
    rolled_lo = rolled_lo.droplevel(0)
    rolled_lo = rolled_lo.to_frame(name="leftovers_mean_7d")
    rolled_lo["nm_id"] = train_sorted["nm_id"].values
    rolled_lo["dt"] = train_sorted["dt"].values
    rolling_features.append(rolled_lo.reset_index(drop=True))

    # Days since last sale (vectorized)
    train_sorted["had_sale"] = (train_sorted["qty"] > 0).astype(int)
    train_sorted["_sale_date"] = train_sorted["dt"].where(train_sorted["had_sale"] == 1)
    train_sorted["_last_sale_date"] = (
        train_sorted.groupby("nm_id")["_sale_date"].ffill()
    )
    train_sorted["_last_sale_date_shifted"] = (
        train_sorted.groupby("nm_id")["_last_sale_date"].shift(1)
    )
    train_sorted["days_since_last_sale"] = (
        (train_sorted["dt"] - train_sorted["_last_sale_date_shifted"]).dt.days
    )
    train_sorted["days_since_last_sale"] = train_sorted["days_since_last_sale"].fillna(999).astype(int)

    days_since_df = train_sorted[["nm_id", "dt", "days_since_last_sale"]].copy()
    rolling_features.append(days_since_df)

    # Streak of consecutive zeros (vectorized)
    train_sorted["_sale_flag"] = (train_sorted["qty"] > 0).astype(int)
    train_sorted["_streak_group"] = train_sorted.groupby("nm_id")["_sale_flag"].cumsum()
    train_sorted["zero_streak"] = train_sorted.groupby(["nm_id", "_streak_group"]).cumcount()

    streak_df = train_sorted[["nm_id", "dt", "zero_streak"]].copy()
    rolling_features.append(streak_df)

    # Clean up temporary columns
    train_sorted.drop(
        columns=["_sale_date", "_last_sale_date", "_last_sale_date_shifted",
                 "_sale_flag", "_streak_group"],
        inplace=True,
    )

    # Merge all rolling features into a single lag lookup table
    lag_table = rolling_features[0]
    for feat_df in rolling_features[1:]:
        lag_table = lag_table.merge(feat_df, on=["nm_id", "dt"], how="outer")

    # Shift: for each item, the lag features at date T should use the values computed at T-1
    lag_table = lag_table.sort_values(["nm_id", "dt"])
    shift_cols = [c for c in lag_table.columns if c not in ["nm_id", "dt"]]
    lag_table[shift_cols] = lag_table.groupby("nm_id")[shift_cols].shift(1)

    # For non-train data: take the last available lag features from train
    last_train_lags = lag_table.groupby("nm_id").last().reset_index()
    # Remember the last train date per item before dropping dt
    last_train_dates = train_sorted.groupby("nm_id")["dt"].max().reset_index()
    last_train_dates.columns = ["nm_id", "_last_train_dt"]
    last_train_lags = last_train_lags.drop("dt", axis=1)

    # Merge lag features into df
    if is_train:
        # Training data — merge on (nm_id, dt) for exact per-day lag values
        df = df.merge(lag_table, on=["nm_id", "dt"], how="left")
    else:
        # Non-train data (tuning/val/test) — use last available lags from train,
        # then adjust temporal features by offset from last train date
        df = df.merge(last_train_lags, on="nm_id", how="left")
        df = df.merge(last_train_dates, on="nm_id", how="left")

        # Offset = how many days this row is past the last train date
        offset = (df["dt"] - df["_last_train_dt"]).dt.days.clip(lower=0)

        # days_since_last_sale grows linearly with each day (no sales observed in test)
        df["days_since_last_sale"] = df["days_since_last_sale"] + offset
        # zero_streak grows linearly (assuming no sales in the gap, which holds for ~87%)
        df["zero_streak"] = df["zero_streak"] + offset

        df = df.drop(columns=["_last_train_dt"])

    return df


def add_promo_features(df: pd.DataFrame, train_df: pd.DataFrame,
                       is_train: bool = True) -> pd.DataFrame:
    """Add promo-related features."""
    df = df.copy()

    # Promo days in last 7 days (from train)
    train_sorted = train_df.sort_values(["nm_id", "dt"]).reset_index(drop=True)
    promo_7d = (
        train_sorted.groupby("nm_id")["is_promo"]
        .rolling(7, min_periods=1)
        .sum()
    )
    promo_7d = promo_7d.droplevel(0)
    promo_7d = promo_7d.to_frame(name="promo_days_7d")
    promo_7d["nm_id"] = train_sorted["nm_id"].values
    promo_7d["dt"] = train_sorted["dt"].values
    promo_7d = promo_7d.reset_index(drop=True)

    # Shift by 1
    promo_7d = promo_7d.sort_values(["nm_id", "dt"])
    promo_7d["promo_days_7d"] = promo_7d.groupby("nm_id")["promo_days_7d"].shift(1)

    if is_train:
        df = df.merge(promo_7d, on=["nm_id", "dt"], how="left")
    else:
        last_promo = promo_7d.groupby("nm_id")["promo_days_7d"].last().reset_index()
        df = df.merge(last_promo, on="nm_id", how="left")

    return df


# ============================================================
# Main pipeline
# ============================================================

def build_features(df: pd.DataFrame, train_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: DataFrame to add features to (train or test/tuning/val)
        train_df: Full training DataFrame (used for computing lag/item features)
        is_train: Whether df is the actual training split (True only for
                  the main training data; False for tuning/val/test)
    """
    print("  Adding calendar features...")
    df = add_calendar_features(df)

    print("  Adding item static features...")
    df = add_item_static_features(df, train_df)

    print("  Adding price features...")
    df = add_price_features(df)

    print("  Adding leftovers features...")
    df = add_leftovers_features(df)

    print("  Adding forward delta features...")
    df = add_forward_delta_features(df)

    print("  Adding price change & backward delta features...")
    df = add_price_change_features(df, train_df, is_train=is_train)

    print("  Adding sell rate features...")
    df = add_sell_rate_features(df, train_df)

    print("  Adding lag features (this may take a while)...")
    df = add_lag_features(df, train_df, is_train=is_train)

    print("  Adding promo features...")
    df = add_promo_features(df, train_df, is_train=is_train)

    print("  Adding market features...")
    df = add_market_features(df)

    print("  Adding interaction features...")
    df = add_interaction_features(df)

    return df


# Feature columns used for modeling (excluding nm_id, dt, qty)
FEATURE_COLS = [
    # Original features
    "price", "is_promo", "prev_leftovers", "sneakers_google_trends",
    # Calendar
    "day_of_week", "day_of_month", "week_of_year", "month", "is_weekend",
    "is_holiday", "days_to_holiday",
    # Item static
    "item_total_qty", "item_mean_qty", "item_std_qty", "item_max_qty",
    "item_pct_nonzero", "item_mean_price", "item_std_price",
    "item_min_price", "item_max_price", "item_mean_leftovers",
    "item_n_days", "item_pct_promo", "item_price_range_pct", "item_cv_qty",
    # Price features
    "price_vs_item_mean", "price_vs_item_min", "price_vs_item_max",
    "price_discount_pct",
    # Leftovers features
    "log_leftovers", "leftovers_vs_item_mean",
    # Forward delta (KEY — corr ~0.52 with qty)
    "fwd_delta_lo", "fwd_delta_lo_raw",
    # Backward delta & price changes
    "bwd_delta_lo", "price_change_1d", "price_change_7d",
    "price_dropped", "price_dropped_big",
    # Sell rate
    "item_sell_rate", "expected_qty",
    # Lag features
    "qty_mean_7d", "qty_sum_7d", "qty_max_7d", "qty_std_7d",
    "qty_mean_14d", "qty_sum_14d", "qty_max_14d", "qty_std_14d",
    "qty_mean_30d", "qty_sum_30d", "qty_max_30d", "qty_std_30d",
    "pct_nonzero_7d", "pct_nonzero_14d", "pct_nonzero_30d",
    "price_mean_7d", "price_mean_14d",
    "leftovers_mean_7d",
    "days_since_last_sale", "zero_streak",
    # Promo features
    "promo_days_7d",
    # Market features
    "market_pct_promo", "market_mean_price", "market_mean_leftovers",
    "price_vs_market",
    # Interaction features
    "promo_and_drop", "promo_x_discount", "leftovers_x_promo", "leftovers_x_pct_nz",
]


if __name__ == "__main__":
    # Quick test
    train = pd.read_parquet("train.parquet")
    test = pd.read_parquet("test.parquet")
    train["dt"] = pd.to_datetime(train["dt"])
    test["dt"] = pd.to_datetime(test["dt"])

    print("Building train features...")
    train_feat = build_features(train, train, is_train=True)
    print(f"Train features shape: {train_feat.shape}")
    print(f"NaN counts:\n{train_feat[FEATURE_COLS].isna().sum()}")

    print("\nBuilding test features...")
    test_feat = build_features(test, train, is_train=False)
    print(f"Test features shape: {test_feat.shape}")
    print(f"NaN counts:\n{test_feat[FEATURE_COLS].isna().sum()}")