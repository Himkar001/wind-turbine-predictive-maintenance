"""
Phase 6 — Pipeline Service (Singleton)
=======================================
Loads PredictiveMaintenancePipeline once at API startup and exposes it
for all prediction endpoints. Avoids reloading heavy ML models per request.

Usage:
    from src.api.services.pipeline_service import get_pipeline, predict_from_snapshot

At startup:
    await initialise_pipeline()
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton state ───────────────────────────────────────────────────────────
_pipeline = None
_models_ready: bool = False
_load_error: Optional[str] = None
_loaded_at: Optional[str] = None


def get_pipeline():
    """Return the loaded pipeline singleton. Raises if not yet loaded."""
    if not _models_ready:
        raise RuntimeError("Pipeline not loaded yet — call initialise_pipeline() at startup")
    return _pipeline


def is_models_ready() -> bool:
    return _models_ready


def get_load_error() -> Optional[str]:
    return _load_error


def get_loaded_at() -> Optional[str]:
    return _loaded_at


def initialise_pipeline() -> bool:
    """
    Load the PredictiveMaintenancePipeline singleton.
    Called once from FastAPI startup event.
    Returns True on success, False on failure.
    """
    global _pipeline, _models_ready, _load_error, _loaded_at

    logger.info("[PipelineService] Loading ML models …")
    t0 = time.time()

    try:
        from src.models.model_pipeline import PredictiveMaintenancePipeline
        pipeline = PredictiveMaintenancePipeline()
        pipeline.load_models()
        _pipeline = pipeline
        _models_ready = True
        _load_error = None
        elapsed = time.time() - t0
        from datetime import datetime, timezone
        _loaded_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"[PipelineService] Models loaded in {elapsed:.1f}s")
        return True

    except Exception as exc:
        _load_error = str(exc)
        _models_ready = False
        logger.error(f"[PipelineService] Model load failed: {exc}")
        return False


# ── Inference helpers ─────────────────────────────────────────────────────────

# Canonical feature columns expected by the model pipeline
# (matches fault_classifier.feature_cols — 87 features)
_SENSOR_FEATURE_COLS = [
    "bearing_temp_1", "bearing_temp_2", "bearing_temp_3", "bearing_temp_4",
    "generator_temp", "generator_bearing_temp", "generator_stator_temp",
    "gearbox_temp", "gearbox_oil_temp", "gearbox_bearing_temp",
    "rotor_speed", "generator_speed", "rotor_torque",
    "wind_speed", "wind_direction", "wind_turbulence_intensity",
    "power_output", "reactive_power", "power_factor", "grid_frequency",
    "vibration_nacelle_x", "vibration_nacelle_y", "vibration_nacelle_z",
    "vibration_tower_x", "vibration_tower_y",
    "oil_pressure", "oil_level", "oil_viscosity",
    "blade_pitch_1", "blade_pitch_2", "blade_pitch_3",
    "yaw_angle", "yaw_error",
    "ambient_temp", "humidity", "air_pressure",
    "hydraulic_pressure", "coolant_temp",
    "voltage_l1", "voltage_l2", "voltage_l3",
    "current_l1", "current_l2", "current_l3",
    # rolling/derived features (will be zero-filled if missing)
    "bearing_temp_1_rolling_mean", "bearing_temp_1_rolling_std",
    "bearing_temp_2_rolling_mean", "bearing_temp_2_rolling_std",
    "generator_temp_rolling_mean", "generator_temp_rolling_std",
    "gearbox_temp_rolling_mean", "gearbox_temp_rolling_std",
    "rotor_speed_rolling_mean", "rotor_speed_rolling_std",
    "power_output_rolling_mean", "power_output_rolling_std",
    "vibration_nacelle_x_rolling_mean", "vibration_nacelle_x_rolling_std",
    "vibration_nacelle_y_rolling_mean", "vibration_nacelle_y_rolling_std",
    "oil_pressure_rolling_mean", "oil_pressure_rolling_std",
    "bearing_temp_range", "vibration_magnitude",
    "power_wind_ratio", "speed_ratio",
    "temp_differential_bearing", "temp_differential_gearbox",
    "hour", "day_of_week", "month",
    "sin_hour", "cos_hour", "sin_day", "cos_day",
    "lag_bearing_temp_1", "lag_generator_temp", "lag_power_output",
    "lag_vibration_nacelle_x", "lag_rotor_speed",
    "is_anomaly", "anomaly_score",
    "residual_bearing_temp_1", "residual_generator_temp",
    "residual_power_output", "residual_vibration_nacelle_x",
    "norm_bearing_temp_1", "norm_generator_temp", "norm_power_output",
]


def _snapshot_to_dataframe(snapshot: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert an arbitrary sensor snapshot dict to a single-row DataFrame
    that matches the model pipeline's expected feature schema.

    Missing columns are filled with 0.0 (safe default for normalised features).
    """
    # Start with zeros for all expected features
    row = {col: 0.0 for col in _SENSOR_FEATURE_COLS}

    # Override with whatever the caller provided
    for k, v in snapshot.items():
        if k in row:
            row[k] = float(v)

    df = pd.DataFrame([row])

    # Ensure all columns exist (paranoia check)
    for col in _SENSOR_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df


def predict_from_snapshot(
    turbine_id: str,
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run inference on a single sensor snapshot dict.

    Returns a dict with:
        turbine_id, anomaly_score, is_anomaly,
        fault_type, fault_confidence, rul_days, rul_urgency,
        risk_score, risk_tier
    """
    pipeline = get_pipeline()

    df = _snapshot_to_dataframe(snapshot)

    result = pipeline.predict_single(df)

    # Map pipeline output keys → API response keys
    return {
        "turbine_id":       turbine_id,
        "anomaly_score":    round(result.get("anomaly_score", 0.0), 4),
        "is_anomaly":       bool(result.get("is_anomaly_prediction", 0)),
        "fault_type":       result.get("predicted_fault_type", "unknown"),
        "fault_confidence": round(result.get("fault_confidence", 0.0), 4),
        "rul_days":         round(result.get("predicted_rul_days", 30.0), 2),
        "rul_urgency":      result.get("rul_urgency", "low"),
        "risk_score":       round(result.get("overall_risk_score", 0.0), 4),
    }
