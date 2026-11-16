"""Serve estimates from the saved EAF LightGBM model."""

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException


ARTIFACT_DIR = Path(
    os.environ.get("EAF_ARTIFACTS_DIR", Path(__file__).resolve().parent / "artifacts")
)
METADATA_FIELDS = {"HEATID", "MW", "", "Unnamed: 0"}

app = FastAPI(title="Electric arc furnace energy estimate")


@lru_cache(maxsize=1)
def read_artifacts(
    directory: Path, file_versions: tuple[tuple[int, int], ...]
) -> tuple[lgb.Booster, dict]:
    # The file versions are part of the cache key. Retraining replaces the files.
    model_bytes = (directory / "lightgbm_model.txt").read_bytes()
    metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(model_bytes).hexdigest() != metadata["model_sha256"]:
        raise ValueError("The model and metadata come from different training runs.")

    model = lgb.Booster(model_str=model_bytes.decode("utf-8"))
    names = metadata["feature_names"]
    defaults = metadata["defaults"]
    if (
        model.feature_name() != metadata["model_feature_names"]
        or len(names) != model.num_feature()
        or len(set(names)) != len(names)
        or set(names) != set(defaults)
        or not all(isinstance(name, str) for name in names)
        or not all(math.isfinite(defaults[name]) for name in names)
        or not isinstance(metadata["target_unit"], str)
    ):
        raise ValueError("The model and metadata have different or invalid input fields.")
    return model, metadata


def get_artifacts() -> tuple[lgb.Booster, dict]:
    try:
        file_stats = [
            (ARTIFACT_DIR / name).stat()
            for name in ("lightgbm_model.txt", "model_metadata.json")
        ]
        versions = tuple((stat.st_mtime_ns, stat.st_size) for stat in file_stats)
        return read_artifacts(ARTIFACT_DIR, versions)
    except FileNotFoundError as error:
        name = Path(error.filename).name
        raise HTTPException(503, f"Missing {name}. Run train.py first.") from error
    except (OSError, ValueError, KeyError, TypeError, OverflowError, lgb.basic.LightGBMError) as error:
        raise HTTPException(503, f"Cannot load the prediction files. {error}") from error


def estimate(model: lgb.Booster, values: list[float]) -> float:
    try:
        prediction = float(model.predict(np.array([values]), num_threads=1)[0])
    except lgb.basic.LightGBMError as error:
        raise HTTPException(503, "The saved model cannot make a prediction.") from error
    if not math.isfinite(prediction):
        raise HTTPException(503, "The saved model returned a non-finite prediction.")
    return prediction


@app.get("/test")
async def test_service() -> dict:
    model, metadata = get_artifacts()
    defaults = metadata["defaults"]
    estimate(model, [defaults[name] for name in metadata["feature_names"]])
    return {"status": "ready"}


@app.post("/predict")
async def predict(payload: dict[str, Any]) -> dict:
    model, metadata = get_artifacts()
    defaults = metadata["defaults"]
    unknown = sorted(payload.keys() - defaults.keys() - METADATA_FIELDS)
    if unknown:
        raise HTTPException(422, {"unknown_fields": unknown})

    values = []
    defaulted = []
    for name in metadata["feature_names"]:
        if name not in payload:
            values.append(defaults[name])
            defaulted.append(name)
            continue

        value = payload[name]
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise HTTPException(422, f"{name} must be a finite number.")
        values.append(float(value))

    return {
        "energy_mwh": estimate(model, values),
        "unit": metadata["target_unit"],
        "defaulted_fields": defaulted,
    }
