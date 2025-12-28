from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rohlik.constants import (
    CALENDAR_COLS,
    CALENDAR_DTYPES,
    INVENTORY_COLS,
    INVENTORY_DTYPES,
    SALES_DTYPES,
    SALES_TEST_COLS,
    SALES_TRAIN_COLS,
    DISCOUNT_COLS,
)


def read_csv(
    path: Path,
    usecols: list[str] | None = None,
    dtype: dict[str, str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        nrows=nrows,
    )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


@dataclass(frozen=True)
class RawData:
    train: pd.DataFrame
    test: pd.DataFrame
    calendar: pd.DataFrame
    inventory: pd.DataFrame
    weights: pd.DataFrame


def load_raw_data(data_dir: Path) -> RawData:
    train = read_csv(
        data_dir / "sales_train.csv",
        usecols=SALES_TRAIN_COLS,
        dtype=SALES_DTYPES,
    )
    test = read_csv(
        data_dir / "sales_test.csv",
        usecols=SALES_TEST_COLS,
        dtype=SALES_DTYPES,
    )
    calendar = read_csv(
        data_dir / "calendar.csv",
        usecols=CALENDAR_COLS,
        dtype=CALENDAR_DTYPES,
    )
    inventory = read_csv(
        data_dir / "inventory.csv",
        usecols=INVENTORY_COLS,
        dtype=INVENTORY_DTYPES,
    )
    weights = read_csv(
        data_dir / "test_weights.csv",
        usecols=["unique_id", "weight"],
        dtype={"unique_id": "int32", "weight": "float32"},
    )

    return RawData(train=train, test=test, calendar=calendar, inventory=inventory, weights=weights)


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


def prepare_frame(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    inventory: pd.DataFrame,
    weights: pd.DataFrame | None = None,
) -> pd.DataFrame:

    df = sales.copy()

    if weights is not None:
        df = df.merge(weights, on="unique_id", how="left")

    df = df.merge(
        inventory.drop(columns=["warehouse"]),
        on="unique_id",
        how="left",
    )

    df = df.merge(calendar, on=["date", "warehouse"], how="left")

    # Заполним holiday_name
    df["holiday_name"] = df["holiday_name"].cat.add_categories(["NONE"]).fillna("NONE")

    df = add_date_features(df)
    df = add_discount_features(df)
    df["log1p_sell_price_main"] = np.log1p(df["sell_price_main"]).astype("float32")

    # Явно типизируем идентификаторы как категориальные (для моделей)
    df["unique_id"] = df["unique_id"].astype("int32").astype("category")
    df["product_unique_id"] = df["product_unique_id"].astype("category")

    return df


@dataclass(frozen=True)
class SplitInfo:
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    horizon_days: int
    train_rows: int
    val_rows: int


def time_holdout_split(df: pd.DataFrame, horizon_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    train_end = df["date"].max()
    val_start = train_end - pd.Timedelta(days=horizon_days - 1)
    is_val = df["date"] >= val_start

    train_part = df.loc[~is_val].copy()
    val_part = df.loc[is_val].copy()

    info = SplitInfo(
        train_end=train_end,
        val_start=val_start,
        horizon_days=horizon_days,
        train_rows=len(train_part),
        val_rows=len(val_part),
    )
    return train_part, val_part, info
