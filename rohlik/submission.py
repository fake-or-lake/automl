from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rohlik.metrics import wmae


def make_competition_id(df: pd.DataFrame) -> pd.Series:
    return df["unique_id"].astype(str) + "_" + df["date"].dt.strftime("%Y-%m-%d")


def save_submission(df_test: pd.DataFrame, pred: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.DataFrame(
        {
            "id": make_competition_id(df_test),
            "sales_hat": pred,
        }
    )
    sub.to_csv(path, index=False)
    return path


def save_predictions_for_config(
    config_name: str,
    output_dir: Path,
    val_df: pd.DataFrame,
    val_pred: np.ndarray,
    test_df: pd.DataFrame,
    test_pred: np.ndarray,
) -> tuple[float, Path, Path]:

    val_pred_arr = np.asarray(val_pred, dtype=np.float64)
    test_pred_arr = np.asarray(test_pred, dtype=np.float64)

    val_pred_clipped = np.clip(val_pred_arr, 0.0, None)
    test_pred_clipped = np.clip(test_pred_arr, 0.0, None)

    score = wmae(
        y_true=val_df["sales"].to_numpy(dtype=np.float64),
        y_pred=val_pred_clipped,
        weight=val_df["weight"].to_numpy(dtype=np.float64),
    )

    pred_val_path = output_dir / f"pred_last14_{config_name}.csv"
    sub_path = output_dir / f"submission_{config_name}.csv"

    save_submission(df_test=val_df[["unique_id", "date"]], pred=val_pred_clipped, path=pred_val_path)
    save_submission(df_test=test_df[["unique_id", "date"]], pred=test_pred_clipped, path=sub_path)

    return score, pred_val_path, sub_path



