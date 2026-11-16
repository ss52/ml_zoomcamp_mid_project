"""Train LightGBM on the prepared EAF data and save the serving artifacts."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "prepared" / "df_combined_cleaned.csv"
DEFAULT_ARTIFACTS = PROJECT_DIR / "artifacts"
SEED = 42
TARGET_UNIT = "MWh/heat (assumed)"


def score(actual, predicted) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
        "r2": r2_score(actual, predicted),
    }


def save_artifacts(model: lgb.Booster, metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as staging_dir:
        staging = Path(staging_dir)
        model_path = staging / "lightgbm_model.txt"
        model.save_model(model_path)
        metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata_path = staging / "model_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        model_path.replace(output_dir / model_path.name)
        metadata_path.replace(output_dir / metadata_path.name)


def train(data_path: Path, output_dir: Path) -> dict:
    heats = pd.read_csv(data_path)
    target = heats.pop("MW")
    predictors = heats.drop(columns=["HEATID", "Unnamed: 0"], errors="ignore")
    feature_names = predictors.columns.to_list()

    # The preparation script puts the individual material masses before Scrap.
    material_columns = [*feature_names[:feature_names.index("Scrap")], "Scrap", "OBM"]
    model_names = [re.sub(r"[^A-Za-z0-9_]+", "_", name) for name in feature_names]
    if len(set(model_names)) != len(model_names):
        raise ValueError("Two input columns have the same LightGBM column name.")

    fit_x, test_x, fit_y, test_y = train_test_split(
        predictors, target, test_size=0.2, random_state=SEED, shuffle=True
    )
    train_x, val_x, train_y, val_y = train_test_split(
        fit_x, fit_y, test_size=0.25, random_state=SEED, shuffle=True
    )
    defaults = train_x.median()
    material_medians = defaults.loc[material_columns]
    defaults.loc[material_columns] = material_medians.where(material_medians >= 1000, 0)

    ridge = make_pipeline(StandardScaler(), RidgeCV(cv=5))
    ridge.fit(train_x, train_y)

    name_map = dict(zip(feature_names, model_names, strict=True))
    named_train = train_x.rename(columns=name_map)
    named_val = val_x.rename(columns=name_map)

    tuning_results = []
    best_model = None
    best_rmse = float("inf")
    best_parameters = {}
    configurations = [(leaves, rounds) for leaves in (15, 31, 63) for rounds in (150, 400)]
    configurations.append((31, 1000))  # Also compare the notebook configuration.
    for leaves, rounds in configurations:
        parameters = {"num_leaves": leaves, "n_estimators": rounds}
        candidate = lgb.LGBMRegressor(
            **parameters,
            learning_rate=0.05,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=2,
            verbosity=-1,
        )
        candidate.fit(named_train, train_y)
        rmse = root_mean_squared_error(val_y, candidate.predict(named_val))
        tuning_results.append({**parameters, "validation_rmse": rmse})
        if rmse < best_rmse:
            best_model = candidate
            best_rmse = rmse
            best_parameters = parameters

    def evaluate(inputs: pd.DataFrame, actual: pd.Series) -> dict:
        return {
            "Training median": score(actual, np.full(len(actual), train_y.median())),
            "Statistical estimate": score(
                actual, inputs["stat_MW"] * inputs["Calc_tap_mass"] / 1000
            ),
            "Ridge": score(actual, ridge.predict(inputs)),
            "LightGBM": score(actual, best_model.predict(inputs.rename(columns=name_map))),
        }

    results = {
        "heats": {"train": len(train_x), "validation": len(val_x), "test": len(test_x)},
        "unit": TARGET_UNIT,
        "lightgbm_parameters": best_parameters,
        "tuning": tuning_results,
        "validation": evaluate(val_x, val_y),
        "test": evaluate(test_x, test_y),
    }
    save_artifacts(
        best_model.booster_,
        {
            "feature_names": feature_names,
            "model_feature_names": model_names,
            "defaults": defaults.to_dict(),
            "target_unit": TARGET_UNIT,
        },
        output_dir,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Prepared CSV file")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Missing {args.data}. Run prepare_data.py first.")
    print(json.dumps(train(args.data, args.artifacts_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
