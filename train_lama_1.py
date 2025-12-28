from __future__ import annotations

import logging
import os

import numpy as np

from rohlik.config import Config
from rohlik.constants import FEATURES_DROP_ALWAYS
from rohlik.data import load_raw_data, prepare_frame, time_holdout_split
from rohlik.submission import save_predictions_for_config

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

logger = logging.getLogger("train_lama_1")

TIMEOUT_SEC = 60 * 60
N_JOBS = 8


def fit_lama(train_df, random_state: int) -> TabularAutoML:
    roles = {
        "target": "sales",
        "drop": FEATURES_DROP_ALWAYS,
        "weights": "weight",
    }

    task = Task("reg", metric="mae")
    automl = TabularAutoML(
        task=task,
        timeout=TIMEOUT_SEC,
        cpu_limit=N_JOBS,
        reader_params={"random_state": random_state, "n_jobs": N_JOBS},
    )

    _ = automl.fit_predict(train_df, roles=roles, verbose=1)
    return automl


def predict_lama(automl, df) -> np.ndarray:
    feat = df.drop(columns=["sales", "weight"] + FEATURES_DROP_ALWAYS, errors="ignore")
    pred = automl.predict(feat).data[:, 0]
    return pred


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = Config()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DATA_DIR=%s", cfg.data_dir.resolve())
    logger.info("OUTPUT_DIR=%s", cfg.output_dir.resolve())
    raw = load_raw_data(cfg.data_dir)
    train = prepare_frame(raw.train, calendar=raw.calendar, inventory=raw.inventory, weights=raw.weights)
    test = prepare_frame(raw.test, calendar=raw.calendar, inventory=raw.inventory, weights=None)

    train = train.dropna(subset=["sales"]).reset_index(drop=True)

    train_part, val_part, info = time_holdout_split(train, horizon_days=cfg.horizon_days)
    logger.info("Split: train_end=%s val_start=%s train_rows=%s val_rows=%s",
                info.train_end.date(), info.val_start.date(), info.train_rows, info.val_rows)

    logger.info("Fit LAMA STRONG on train_part (without last 14 days)...")
    automl = fit_lama(
        train_df=train_part,
        random_state=cfg.random_state,
    )

    val_pred = predict_lama(automl, val_part)
    test_pred = predict_lama(automl, test)

    score, pred_val_path, sub_path = save_predictions_for_config(
        config_name="lama_1",
        output_dir=cfg.output_dir,
        val_df=val_part,
        val_pred=val_pred,
        test_df=test,
        test_pred=test_pred,
    )
    logger.info("Outer holdout WMAE: %.6f", score)
    logger.info("Saved: %s", pred_val_path.resolve())
    logger.info("Saved: %s", sub_path.resolve())


if __name__ == "__main__":
    main()


