from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetStatsFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, recent_window_days: int = 28):
        self.recent_window_days = recent_window_days
        self.column_order = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series):
        df = X.copy().reset_index(drop=True)
        y_arr = np.asarray(y, dtype=np.float64)

        self.global_mean_ = float(np.nanmean(y_arr))

        self.total_orders_median_by_warehouse_ = df.groupby("warehouse", observed=True)["total_orders"].median().to_dict()
        self.total_orders_global_median_ = float(np.nanmedian(df["total_orders"]))

        df["__y__"] = y_arr

        self.uid_mean_ = df.groupby(["unique_id"], observed=True)["__y__"].mean().rename("te_uid_mean")
        self.wh_mean_ = df.groupby(["warehouse"], observed=True)["__y__"].mean().rename("te_wh_mean")

        self.uid_dow_mean_ = (
            df.groupby(["unique_id", "dayofweek"], observed=True)["__y__"].mean().rename("te_uid_dow_mean")
        )
        self.wh_dow_mean_ = (
            df.groupby(["warehouse", "dayofweek"], observed=True)["__y__"].mean().rename("te_wh_dow_mean")
        )

        dates = pd.to_datetime(df["date"])
        last_date = dates.max()
        start_recent = last_date - pd.Timedelta(days=self.recent_window_days - 1)
        recent = df[dates >= start_recent]
        self.uid_recent_mean_ = (
            recent.groupby(["unique_id"], observed=True)["__y__"]
            .mean()
            .rename(f"te_uid_mean_last_{self.recent_window_days}d")
        )

        df.drop(columns=["__y__"], inplace=True)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df["total_orders"] = df["total_orders"].fillna(df["warehouse"].map(self.total_orders_median_by_warehouse_))
        df["total_orders"] = df["total_orders"].fillna(self.total_orders_global_median_)

        df = df.join(self.uid_mean_, on=["unique_id"])
        df = df.join(self.wh_mean_, on=["warehouse"])
        df["te_uid_mean"] = df["te_uid_mean"].fillna(self.global_mean_)
        df["te_wh_mean"] = df["te_wh_mean"].fillna(self.global_mean_)

        df = df.join(self.uid_dow_mean_, on=["unique_id", "dayofweek"])
        df["te_uid_dow_mean"] = df["te_uid_dow_mean"].fillna(df["te_uid_mean"])

        df = df.join(self.wh_dow_mean_, on=["warehouse", "dayofweek"])
        df["te_wh_dow_mean"] = df["te_wh_dow_mean"].fillna(df["te_wh_mean"])

        df = df.join(self.uid_recent_mean_, on=["unique_id"])
        col = f"te_uid_mean_last_{self.recent_window_days}d"
        df[col] = df[col].fillna(df["te_uid_mean"])

        out = df.select_dtypes(include=[np.number]).copy()
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        if self.column_order is None:
            self.column_order = out.columns.tolist()

        out = out.reindex(columns=self.column_order, fill_value=0.0)
        return out



