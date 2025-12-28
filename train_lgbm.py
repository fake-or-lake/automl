from __future__ import annotations

import logging
import json

import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

from rohlik.config import Config
from rohlik.constants import FEATURES_DROP_ALWAYS
from rohlik.data import load_raw_data, prepare_frame, time_holdout_split
from rohlik.metrics import wmae
from rohlik.models.target_stats import TargetStatsFeaturizer
from rohlik.submission import save_predictions_for_config


logger = logging.getLogger("train_lgbm")

RECENT_WINDOW_DAYS = 28

INNER_HORIZON_DAYS = 14
OPTUNA_TRIALS = 30
OPTUNA_TIMEOUT_SEC = 3600

N_JOBS = 8
N_ESTIMATORS = 10_000
EARLY_STOPPING_ROUNDS = 200

TUNE_SAMPLE_FRAC = 0.1
TUNE_N_ESTIMATORS = 3_000
TUNE_EARLY_STOPPING_ROUNDS = 100
LGBM_LOG_EVAL_PERIOD = 200


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["sales", "weight"] + FEATURES_DROP_ALWAYS
    return df.drop(columns=drop_cols, errors="ignore")


def train_lgbm_with_te(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    w_val: np.ndarray,
    recent_window_days: int,
    model_params: dict,
    n_estimators: int,
    early_stopping_rounds: int,
):
    logger.info("TargetStatsFeaturizer.fit() ...")
    fe = TargetStatsFeaturizer(recent_window_days=recent_window_days)
    fe.fit(X_train, y_train)

    X_train_fe = fe.transform(X_train)
    X_val_fe = fe.transform(X_val)

    logger.info("LGBMRegressor.fit() ... n_estimators=%s early_stopping_rounds=%s", n_estimators, early_stopping_rounds)
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        **model_params,
    )

    model.fit(
        X_train_fe,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val_fe, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True, min_delta=0.01),
            lgb.log_evaluation(period=LGBM_LOG_EVAL_PERIOD),
        ],
    )

    best_iter = int(getattr(model, "best_iteration_", n_estimators) or n_estimators)
    val_pred = model.predict(X_val_fe)
    return model, fe, best_iter, val_pred


def default_params() -> dict:
    return {
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1e-8,
        "reg_lambda": 1e-8,
        "min_split_gain": 0.0,
    }


def tune_params(train_df: pd.DataFrame, seed: int) -> dict:
    inner_train_df, inner_val_df, inner_info = time_holdout_split(train_df, horizon_days=INNER_HORIZON_DAYS)
    logger.info(
        "Inner split (for tuning): train_end=%s val_start=%s train_rows=%s val_rows=%s",
        inner_info.train_end.date(),
        inner_info.val_start.date(),
        inner_info.train_rows,
        inner_info.val_rows,
    )

    inner_train_df = inner_train_df.sample(frac=TUNE_SAMPLE_FRAC, random_state=seed).reset_index(drop=True)
    inner_val_df = inner_val_df.sample(frac=TUNE_SAMPLE_FRAC, random_state=seed).reset_index(drop=True)

    X_train_inner = drop_useless_columns(inner_train_df)
    y_train_inner = inner_train_df["sales"].to_numpy(dtype=np.float64)
    w_train_inner = inner_train_df["weight"].to_numpy(dtype=np.float64)

    X_val_inner = drop_useless_columns(inner_val_df)
    y_val_inner = inner_val_df["sales"].to_numpy(dtype=np.float64)
    w_val_inner = inner_val_df["weight"].to_numpy(dtype=np.float64)

    def objective(trial: "optuna.Trial") -> float:
        logger.info("[Optuna] Trial %s/%s started", trial.number + 1, OPTUNA_TRIALS)
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": seed,
            "n_jobs": N_JOBS,
            "device_type": "gpu",
        }

        _, _, _, pred = train_lgbm_with_te(
            X_train=X_train_inner,
            y_train=y_train_inner,
            w_train=w_train_inner,
            X_val=X_val_inner,
            y_val=y_val_inner,
            w_val=w_val_inner,
            recent_window_days=RECENT_WINDOW_DAYS,
            model_params=params,
            n_estimators=TUNE_N_ESTIMATORS,
            early_stopping_rounds=TUNE_EARLY_STOPPING_ROUNDS,
        )
        pred = np.clip(pred, 0.0, None)
        return wmae(y_true=y_val_inner, y_pred=pred, weight=w_val_inner)

    best_params = default_params(seed=seed)
    logger.info("Optuna tuning started: trials=%s timeout=%s", OPTUNA_TRIALS, OPTUNA_TIMEOUT_SEC)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT_SEC, show_progress_bar=True)

    best_params = {**best_params, **study.best_params}
    """
    Best params from Optuna:
    best_params = {
        "learning_rate": 0.06841286767575318,
        "num_leaves": 409,
        "min_child_samples": 198,
        "subsample": 0.5114899400081168,
        "colsample_bytree": 0.5713518940291988,
        "reg_alpha": 1.0778792775978312,
        "reg_lambda": 0.24642683735423607,
        "min_split_gain": 0.14641387115448795
    }
    """
    logger.info("Optuna best WMAE (inner): %.6f", float(study.best_value))
    logger.info("Optuna best params: %s", best_params)

    return best_params


def fit_lgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict, seed: int) -> tuple[LGBMRegressor, TargetStatsFeaturizer]:
    X_train = drop_useless_columns(train_df)
    y_train = train_df["sales"].to_numpy(dtype=np.float64)
    w_train = train_df["weight"].to_numpy(dtype=np.float64)

    X_val = drop_useless_columns(val_df)
    y_val = val_df["sales"].to_numpy(dtype=np.float64)
    w_val = val_df["weight"].to_numpy(dtype=np.float64)

    model, fe, _best_iter, _ = train_lgbm_with_te(
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val,
        w_val=w_val,
        recent_window_days=RECENT_WINDOW_DAYS,
        model_params={**params, "random_state": seed, "n_jobs": N_JOBS, "device_type": "gpu"},
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    return model, fe


def predict_lgbm(model: LGBMRegressor, fe: TargetStatsFeaturizer, df: pd.DataFrame) -> np.ndarray:
    X = drop_useless_columns(df)
    X_fe = fe.transform(X)
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter:
        return model.predict(X_fe, num_iteration=int(best_iter))
    return model.predict(X_fe)


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

    train_part, val_part, split_info = time_holdout_split(train, horizon_days=cfg.horizon_days)
    logger.info("Split: train_end=%s val_start=%s train_rows=%s val_rows=%s",
                split_info.train_end.date(), split_info.val_start.date(),
                split_info.train_rows, split_info.val_rows)

    best_params = tune_params(train_df=train_part, seed=cfg.random_state)

    params_path = cfg.output_dir / "lgbm_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved best params: %s", params_path.resolve())

    model, fe = fit_lgbm(train_df=train_part, val_df=val_part, params=best_params, seed=cfg.random_state)
    val_pred = predict_lgbm(model=model, fe=fe, df=val_part)
    test_pred = predict_lgbm(model=model, fe=fe, df=test)

    score_outer, pred_val_path, out_path = save_predictions_for_config(
        config_name="lgbm",
        output_dir=cfg.output_dir,
        val_df=val_part,
        val_pred=val_pred,
        test_df=test,
        test_pred=test_pred,
    )
    logger.info("Outer holdout WMAE: %.6f", score_outer)
    logger.info("Saved: %s", pred_val_path.resolve())
    logger.info("Saved: %s", out_path.resolve())


if __name__ == "__main__":
    main()


