from __future__ import annotations

import numpy as np
import pandas as pd

from rohlik.constants import DISCOUNT_COLS


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["day"] = df["date"].dt.day.astype("int8")
    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")
    return df


def add_discount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["discount_max"] = df[DISCOUNT_COLS].max(axis=1).astype("float32")
    df["discount_min"] = df[DISCOUNT_COLS].min(axis=1).astype("float32")
    df["has_discount"] = (df["discount_max"] > 0).astype("int8")
    df["has_negative_discount"] = (df["discount_min"] < 0).astype("int8")

    clipped = df[DISCOUNT_COLS].clip(lower=0.0, upper=1.0)
    df["discount_sum_clipped"] = clipped.sum(axis=1).astype("float32")
    df["discount_mean_clipped"] = clipped.mean(axis=1).astype("float32")
    df["discount_std_clipped"] = clipped.std(axis=1).astype("float32")
    df["discount_nonzero_cnt"] = (clipped > 0).sum(axis=1).astype("int8")
    return df


