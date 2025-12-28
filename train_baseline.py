from __future__ import annotations

import logging

import numpy as np

from rohlik.config import Config
from rohlik.data import read_csv, time_holdout_split
from rohlik.submission import save_predictions_for_config


logger = logging.getLogger("train_baseline")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = Config()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DATA_DIR=%s", cfg.data_dir.resolve())
    logger.info("OUTPUT_DIR=%s", cfg.output_dir.resolve())

    train = read_csv(
        cfg.data_dir / "sales_train.csv",
        usecols=["unique_id", "date", "sales"],
    )
    test = read_csv(
        cfg.data_dir / "sales_test.csv",
        usecols=["unique_id", "date"],
    )
    weights = read_csv(
        cfg.data_dir / "test_weights.csv",
        usecols=["unique_id", "weight"],
        dtype={"unique_id": "int32", "weight": "float32"},
    )

    train = train.merge(weights, on="unique_id", how="left")

    train_part, val_part, info = time_holdout_split(train, horizon_days=cfg.horizon_days)
    logger.info("Split: train_end=%s val_start=%s train_rows=%s val_rows=%s",
                info.train_end.date(), info.val_start.date(), info.train_rows, info.val_rows)

    uid_mean = train_part.groupby("unique_id", observed=True)["sales"].mean()
    global_mean = float(train_part["sales"].mean())

    val_pred = val_part["unique_id"].map(uid_mean).fillna(global_mean).to_numpy(dtype=np.float64)
    test_pred = test["unique_id"].map(uid_mean).fillna(global_mean).to_numpy(dtype=np.float64)

    score, pred_val_path, sub_path = save_predictions_for_config(
        config_name="baseline",
        output_dir=cfg.output_dir,
        val_df=val_part,
        val_pred=val_pred,
        test_df=test,
        test_pred=test_pred,
    )
    logger.info("Holdout WMAE: %.6f", score)
    logger.info("Saved: %s", pred_val_path.resolve())
    logger.info("Saved: %s", sub_path.resolve())


if __name__ == "__main__":
    main()


