from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rohlik.config import Config
from rohlik.data import read_csv, time_holdout_split
from rohlik.submission import save_predictions_for_config

from lightautoml.tasks import Task
from lightautoml.addons.autots.base import AutoTS
from lightautoml.dataset.roles import DatetimeRole


logger = logging.getLogger("train_lama_2")

TRAIN_WINDOW_DAYS = 365
HISTORY = 56
TIMEOUT_SEC = 60 * 60
N_JOBS = 8


def make_dense_df(
    df_sparse: pd.DataFrame,
    unique_ids: np.ndarray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    weights_df: pd.DataFrame | None = None,
) -> pd.DataFrame:

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    grid = pd.MultiIndex.from_product([unique_ids, date_range], names=["unique_id", "date"]).to_frame(index=False)

    merged = grid.merge(weights_df, on=["unique_id"], how="left")
    merged["weight"] = merged["weight"].fillna(1.0).astype("float32")

    merged = merged.merge(df_sparse, on=["unique_id", "date"], how="left")
    merged["sales"] = merged["sales"].fillna(0.0).astype("float32")

    merged["unique_id"] = merged["unique_id"].astype("int32")

    merged = merged.sort_values(["unique_id", "date"]).reset_index(drop=True)
    return merged


def fit_autots(
    train_dense: pd.DataFrame,
    horizon_days: int,
    history: int,
    seed: int,
) -> AutoTS:
    roles = {
        "target": "sales",
        DatetimeRole(base_date=True, seasonality=["y", "m", "d"]): "date",
        "id": "unique_id",
        "weights": "weight",
    }

    reader_params = {
        "random_state": seed,
        "seq_params": {
            "seq0": {
                "case": "next_values",
                "params": {
                    "n_target": int(horizon_days),
                    "history": int(history),
                    "step": 1,
                    "from_last": True,
                    "test_last": True,
                },
            }
        },
    }


    autots = AutoTS(
        Task("multi:reg", greater_is_better=False, metric="mae", loss="mae"),
        reader_params=reader_params,
        general_params={"use_algos": [["linear_l2"]]},
        timeout=TIMEOUT_SEC,
        cpu_limit=N_JOBS,
    )

    _ = autots.fit_predict(train_dense, roles=roles, verbose=1)
    return autots


def predict_autots(autots: AutoTS, train_dense: pd.DataFrame) -> np.ndarray:
    pred, _trend = autots.predict(train_dense)
    return np.asarray(pred, dtype=np.float64)


def predict_from_long(
    df: pd.DataFrame,
    pred_long: pd.DataFrame,
) -> np.ndarray:
    merged = df.merge(pred_long, on=["unique_id", "date"], how="left")
    pred = merged["sales_hat"].to_numpy(dtype=np.float64)
    return pred


def matrix_to_long(
    unique_ids: list[int],
    start_date: pd.Timestamp,
    horizon_days: int,
    pred_matrix: np.ndarray,
) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=horizon_days, freq="D")
    uid_arr = np.repeat(np.asarray(unique_ids, dtype=np.int32), horizon_days)
    date_arr = np.tile(dates.to_numpy(), len(unique_ids))
    pred_arr = np.asarray(pred_matrix, dtype=np.float64).reshape(-1)
    return pd.DataFrame({"unique_id": uid_arr, "date": date_arr, "sales_hat": pred_arr})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = Config()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DATA_DIR=%s", cfg.data_dir.resolve())
    logger.info("OUTPUT_DIR=%s", cfg.output_dir.resolve())

    train = read_csv(cfg.data_dir / "sales_train.csv", usecols=["unique_id", "date", "sales"])
    test = read_csv(cfg.data_dir / "sales_test.csv", usecols=["unique_id", "date"])
    weights = read_csv(
        cfg.data_dir / "test_weights.csv",
        usecols=["unique_id", "weight"],
        dtype={"unique_id": "int32", "weight": "float32"},
    )

    train = train.merge(weights, on="unique_id", how="left")
    train["weight"] = train["weight"].fillna(1.0).astype("float32")
    train = train.dropna(subset=["sales"]).reset_index(drop=True)

    train_part, val_part, split_info = time_holdout_split(train, horizon_days=cfg.horizon_days)
    val_start = split_info.val_start
    train_part_end = split_info.val_start - pd.Timedelta(days=1)
    logger.info(
        "Split: train_end=%s val_start=%s train_rows=%s val_rows=%s",
        split_info.train_end.date(),
        split_info.val_start.date(),
        split_info.train_rows,
        split_info.val_rows,
    )

    uids_val = val_part["unique_id"].unique()
    uids_test = test["unique_id"].unique()
    uids_union = np.unique(np.concatenate([uids_val, uids_test]))
    uids_union_sorted = np.sort(uids_union.astype(int))

    start_window = train_part_end - pd.Timedelta(days=TRAIN_WINDOW_DAYS - 1)
    train_part_window = train_part[(train_part["date"] >= start_window) & (train_part["date"] <= train_part_end)].copy()
    train_part_window = train_part_window[train_part_window["unique_id"].isin(uids_union_sorted)]

    logger.info("AutoTS (VAL): unique_ids=%s window=%s..%s rows_sparse=%s",
                len(uids_union_sorted), start_window.date(), train_part_end.date(), len(train_part_window))

    train_dense = make_dense_df(
        df_sparse=train_part_window[["unique_id", "date", "sales"]],
        unique_ids=uids_union_sorted,
        start_date=start_window,
        end_date=train_part_end,
        weights_df=weights,
    )

    forecast_end = test["date"].max()
    horizon_total = int((forecast_end - val_start).days + 1)
    logger.info("AutoTS horizon_total=%s days (from %s to %s)", horizon_total, val_start.date(), forecast_end.date())

    autots = fit_autots(
        train_dense=train_dense,
        horizon_days=horizon_total,
        history=HISTORY,
        seed=cfg.random_state,
    )
    pred_matrix = predict_autots(autots=autots, train_dense=train_dense)

    pred_long = matrix_to_long(
        unique_ids=uids_union_sorted.tolist(),
        start_date=val_start,
        horizon_days=horizon_total,
        pred_matrix=pred_matrix,
    )

    val_pred = predict_from_long(df=val_part, pred_long=pred_long)
    test_pred = predict_from_long(df=test, pred_long=pred_long)

    score, pred_val_path, sub_path = save_predictions_for_config(
        config_name="lama_2",
        output_dir=cfg.output_dir,
        val_df=val_part,
        val_pred=val_pred,
        test_df=test,
        test_pred=test_pred,
    )
    logger.info("Outer holdout WMAE (AutoTS): %.6f", score)
    logger.info("Saved: %s", pred_val_path.resolve())
    logger.info("Saved: %s", sub_path.resolve())


if __name__ == "__main__":
    main()


