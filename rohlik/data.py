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
)
from rohlik.features import add_date_features, add_discount_features


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



