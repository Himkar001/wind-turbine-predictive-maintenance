"""
risk_scorer.py — Phase 4
========================
Composite Risk Scoring Engine

Fuses three model signals into a single 0→1 risk score per reading,
with turbine-level aggregation for fleet-wide ranking.

Inputs (from model_pipeline.py output):
  - anomaly_score       : float [0,1]  — isolation forest / LSTM AE composite
  - is_anomaly          : bool         — thresholded anomaly flag
  - fault_type          : str          — predicted fault class
  - fault_confidence    : float [0,1]  — max class probability
  - rul_days            : float        — predicted remaining useful life (days)
  - rul_urgency         : str          — critical / high / medium / low

Outputs:
  - risk_score          : float [0,1]  — final composite risk score
  - risk_tier           : str          — CRITICAL / HIGH / MEDIUM / LOW
  - dominant_signal     : str          — which model drove the score (for RCA agent)
  - risk_breakdown      : dict         — per-signal contributions (audit trail)
  - fleet_rank          : int          — turbine rank in fleet (1 = most at risk)

Fusion Logic
------------
The score uses a dynamic-weight blend:

  risk = w_anomaly * S_anomaly
       + w_fault   * S_fault
       + w_rul     * S_rul

Weights adapt when classifier confidence is low (< 0.4):
  - Anomaly weight increases (compensates for uncertain fault label)
  - Fault weight decreases proportionally

RUL urgency bands → urgency multiplier applied *after* blending to
sharpen scores when time-to-failure is genuinely short.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base fusion weights (must sum to 1.0)
BASE_WEIGHTS = {
    "anomaly": 0.40,
    "fault":   0.35,
    "rul":     0.25,
}

# When fault_confidence < LOW_CONF_THRESHOLD, shift weight from fault → anomaly
LOW_CONF_THRESHOLD = 0.40

# Maximum weight transfer allowed from fault → anomaly channel
MAX_WEIGHT_TRANSFER = 0.20

# RUL urgency → scalar sub-score
RUL_URGENCY_SCORES = {
    "critical": 1.00,   # < 2 days
    "high":     0.75,   # 2–7 days
    "medium":   0.45,   # 7–14 days
    "low":      0.10,   # > 14 days
}

# Urgency multiplier: applied post-blend when RUL is critical/high
# Scales score toward 1.0 to ensure short-RUL machines always rank high
RUL_URGENCY_MULTIPLIERS = {
    "critical": 1.30,
    "high":     1.15,
    "medium":   1.00,
    "low":      1.00,
}

# Fault severity map — how dangerous each fault type is independent of confidence
FAULT_SEVERITY = {
    "bearing_failure":      0.90,
    "gearbox_fault":        0.85,
    "generator_overheating": 0.80,
    "electrical_fault":     0.75,
    "oil_leak":             0.65,
    "blade_imbalance":      0.55,
    "maintenance":          0.20,
    "normal":               0.00,
}

# Risk tier thresholds (applied to final clipped [0,1] score)
RISK_TIERS = [
    (0.80, "CRITICAL"),
    (0.60, "HIGH"),
    (0.35, "MEDIUM"),
    (0.00, "LOW"),
]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataclass: per-row risk breakdown (used by RCA agent in Phase 5)
# ---------------------------------------------------------------------------

@dataclass
class RiskBreakdown:
    anomaly_contribution: float
    fault_contribution:   float
    rul_contribution:     float
    anomaly_weight:       float
    fault_weight:         float
    rul_weight:           float
    rul_urgency_mult:     float
    pre_clip_score:       float


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def _compute_anomaly_sub_score(
    anomaly_score: float,
    is_anomaly: bool,
) -> float:
    """
    Normalize the raw anomaly score to [0, 1].

    anomaly_score is already a probability-like float from the detector
    ensemble. We boost it slightly when the hard flag fires to avoid
    borderline cases being dragged below the fault sub-score.
    """
    base = float(np.clip(anomaly_score, 0.0, 1.0))
    if is_anomaly and base < 0.5:
        base = max(base, 0.55)  # floor anomalies at 0.55
    return base


def _compute_fault_sub_score(
    fault_type: str,
    fault_confidence: float,
) -> float:
    """
    Combine fault severity (domain knowledge) with classifier confidence.

    S_fault = severity(fault_type) * confidence

    When fault_type == 'normal', sub-score is 0 regardless of confidence.
    """
    severity = FAULT_SEVERITY.get(str(fault_type).lower(), 0.50)  # default mid
    confidence = float(np.clip(fault_confidence, 0.0, 1.0))
    return severity * confidence


def _compute_rul_sub_score(rul_urgency: str) -> float:
    """Map RUL urgency band to a scalar sub-score."""
    return RUL_URGENCY_SCORES.get(str(rul_urgency).lower(), 0.10)


def _adaptive_weights(fault_confidence: float) -> Tuple[float, float, float]:
    """
    Return (w_anomaly, w_fault, w_rul) with dynamic rebalancing.

    When the classifier is uncertain, transfer up to MAX_WEIGHT_TRANSFER
    from the fault channel to the anomaly channel so the more reliable
    unsupervised signal dominates.
    """
    w_a = BASE_WEIGHTS["anomaly"]
    w_f = BASE_WEIGHTS["fault"]
    w_r = BASE_WEIGHTS["rul"]

    if fault_confidence < LOW_CONF_THRESHOLD:
        deficit = LOW_CONF_THRESHOLD - fault_confidence   # 0 → 0.40
        transfer = min(MAX_WEIGHT_TRANSFER, deficit * (MAX_WEIGHT_TRANSFER / LOW_CONF_THRESHOLD))
        w_a += transfer
        w_f -= transfer
        # Clamp to avoid negatives
        w_f = max(w_f, 0.0)
        # Re-normalise (rul weight stays fixed)
        total = w_a + w_f + w_r
        w_a /= total
        w_f /= total
        w_r /= total

    return w_a, w_f, w_r


def compute_risk_score(
    anomaly_score: float,
    is_anomaly: bool,
    fault_type: str,
    fault_confidence: float,
    rul_days: float,
    rul_urgency: str,
) -> Tuple[float, str, str, RiskBreakdown]:
    """
    Compute the composite risk score for a single reading.

    Returns
    -------
    risk_score     : float [0, 1]
    risk_tier      : str   — CRITICAL / HIGH / MEDIUM / LOW
    dominant_signal: str   — 'anomaly' | 'fault' | 'rul'
    breakdown      : RiskBreakdown
    """
    # 1. Sub-scores
    s_anomaly = _compute_anomaly_sub_score(anomaly_score, is_anomaly)
    s_fault   = _compute_fault_sub_score(fault_type, fault_confidence)
    s_rul     = _compute_rul_sub_score(rul_urgency)

    # 2. Adaptive weights
    w_a, w_f, w_r = _adaptive_weights(fault_confidence)

    # 3. Weighted blend
    c_anomaly = w_a * s_anomaly
    c_fault   = w_f * s_fault
    c_rul     = w_r * s_rul
    blended   = c_anomaly + c_fault + c_rul

    # 4. RUL urgency multiplier (sharpens score for imminent failures)
    mult = RUL_URGENCY_MULTIPLIERS.get(str(rul_urgency).lower(), 1.0)
    pre_clip = blended * mult

    # 5. Clip to [0, 1]
    risk_score = float(np.clip(pre_clip, 0.0, 1.0))

    # 6. Risk tier
    risk_tier = "LOW"
    for threshold, tier in RISK_TIERS:
        if risk_score >= threshold:
            risk_tier = tier
            break

    # 7. Dominant signal (for RCA agent — which model drove this score?)
    contributions = {"anomaly": c_anomaly, "fault": c_fault, "rul": c_rul}
    dominant_signal = max(contributions, key=contributions.get)

    breakdown = RiskBreakdown(
        anomaly_contribution=round(c_anomaly, 4),
        fault_contribution=round(c_fault, 4),
        rul_contribution=round(c_rul, 4),
        anomaly_weight=round(w_a, 4),
        fault_weight=round(w_f, 4),
        rul_weight=round(w_r, 4),
        rul_urgency_mult=mult,
        pre_clip_score=round(pre_clip, 4),
    )

    return risk_score, risk_tier, dominant_signal, breakdown


# ---------------------------------------------------------------------------
# Vectorised batch scoring (DataFrame → DataFrame)
# ---------------------------------------------------------------------------

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply risk scoring to every row of a pipeline output DataFrame.

    Expected columns (from model_pipeline.py):
        anomaly_score, is_anomaly, fault_type, fault_confidence,
        rul_days, rul_urgency

    Returns the original DataFrame augmented with:
        risk_score, risk_tier, dominant_signal, risk_breakdown_json
    """
    required = {
        "anomaly_score", "is_anomaly", "fault_type",
        "fault_confidence", "rul_days", "rul_urgency",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    log.info("Scoring %d rows …", len(df))

    risk_scores      = np.zeros(len(df), dtype=np.float32)
    risk_tiers       = []
    dominant_signals = []
    breakdowns_json  = []

    for i, row in enumerate(df.itertuples(index=False)):
        score, tier, dom, bd = compute_risk_score(
            anomaly_score=float(row.anomaly_score),
            is_anomaly=bool(row.is_anomaly),
            fault_type=str(row.fault_type),
            fault_confidence=float(row.fault_confidence),
            rul_days=float(row.rul_days),
            rul_urgency=str(row.rul_urgency),
        )
        risk_scores[i] = score
        risk_tiers.append(tier)
        dominant_signals.append(dom)
        breakdowns_json.append(json.dumps(asdict(bd)))

    out = df.copy()
    out["risk_score"]         = risk_scores
    out["risk_tier"]          = risk_tiers
    out["dominant_signal"]    = dominant_signals
    out["risk_breakdown_json"] = breakdowns_json

    log.info("Scoring complete")
    return out


# ---------------------------------------------------------------------------
# Fleet-level aggregation
# ---------------------------------------------------------------------------

def aggregate_fleet_risk(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up per-reading risk scores to turbine-level summary.

    Computes:
      - mean_risk_score     : rolling health indicator
      - max_risk_score      : worst-case reading in window
      - p95_risk_score      : 95th-percentile (robust to spikes)
      - critical_count      : number of CRITICAL readings
      - high_count          : number of HIGH readings
      - dominant_fault      : most common non-normal fault type
      - min_rul_days        : most urgent RUL across readings
      - fleet_rank          : rank 1 = most at risk
    """
    if "turbine_id" not in scored_df.columns:
        raise ValueError("DataFrame must contain 'turbine_id' column")

    agg = (
        scored_df
        .groupby("turbine_id")
        .agg(
            mean_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            p95_risk_score=("risk_score", lambda x: x.quantile(0.95)),
            critical_count=("risk_tier", lambda x: (x == "CRITICAL").sum()),
            high_count=("risk_tier", lambda x: (x == "HIGH").sum()),
            min_rul_days=("rul_days", "min"),
            total_readings=("risk_score", "count"),
        )
        .reset_index()
    )

    # Dominant fault: most frequent non-normal, non-maintenance fault
    excl = {"normal", "maintenance"}
    fault_modes = (
        scored_df[~scored_df["fault_type"].str.lower().isin(excl)]
        .groupby("turbine_id")["fault_type"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "none")
        .rename("dominant_fault")
        .reset_index()
    )
    # turbines with no active faults get "none"
    all_turbines = pd.DataFrame({"turbine_id": scored_df["turbine_id"].unique()})
    fault_modes = all_turbines.merge(fault_modes, on="turbine_id", how="left")
    fault_modes["dominant_fault"] = fault_modes["dominant_fault"].fillna("none")
    agg = agg.merge(fault_modes, on="turbine_id")

    # Fleet rank: composite of p95 score + critical count weight
    agg["fleet_rank_score"] = (
        agg["p95_risk_score"] * 0.60
        + (agg["critical_count"] / agg["total_readings"].clip(lower=1)) * 0.25
        + (1.0 / (agg["min_rul_days"].clip(lower=0.1) + 1.0)) * 0.15
    )
    agg["fleet_rank"] = (
        agg["fleet_rank_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    agg = agg.drop(columns=["fleet_rank_score"]).sort_values("fleet_rank")

    # Round floats for readability
    float_cols = ["mean_risk_score", "max_risk_score", "p95_risk_score", "min_rul_days"]
    agg[float_cols] = agg[float_cols].round(4)

    return agg


# ---------------------------------------------------------------------------
# Convenience: risk score summary for a single turbine (used by agents)
# ---------------------------------------------------------------------------

def turbine_risk_summary(
    scored_df: pd.DataFrame,
    turbine_id: str,
    last_n: int = 360,          # last hour of 10-sec readings
) -> Dict:
    """
    Return a JSON-serialisable dict summarising risk for one turbine.
    Designed as the payload handed to the RCA / report agents in Phase 5.
    """
    sub = scored_df[scored_df["turbine_id"] == turbine_id].tail(last_n)
    if sub.empty:
        return {"turbine_id": turbine_id, "error": "no data"}

    latest = sub.iloc[-1]

    # Breakdown of latest reading
    try:
        bd = json.loads(latest["risk_breakdown_json"])
    except Exception:
        bd = {}

    return {
        "turbine_id":         turbine_id,
        "latest_risk_score":  round(float(latest["risk_score"]), 4),
        "risk_tier":          str(latest["risk_tier"]),
        "dominant_signal":    str(latest["dominant_signal"]),
        "fault_type":         str(latest["fault_type"]),
        "fault_confidence":   round(float(latest["fault_confidence"]), 4),
        "rul_days":           round(float(latest["rul_days"]), 2),
        "rul_urgency":        str(latest["rul_urgency"]),
        "anomaly_score":      round(float(latest["anomaly_score"]), 4),
        "is_anomaly":         bool(latest["is_anomaly"]),
        "window_mean_risk":   round(float(sub["risk_score"].mean()), 4),
        "window_max_risk":    round(float(sub["risk_score"].max()), 4),
        "critical_pct":       round(float((sub["risk_tier"] == "CRITICAL").mean()), 4),
        "risk_breakdown":     bd,
    }


# ---------------------------------------------------------------------------
# RiskScorer class (stateless — wraps functions for pipeline import)
# ---------------------------------------------------------------------------

class RiskScorer:
    """
    Stateless risk scoring engine.

    Usage
    -----
    scorer = RiskScorer()
    scored_df     = scorer.score(pipeline_df)
    fleet_summary = scorer.fleet_summary(scored_df)
    turbine_info  = scorer.turbine_summary(scored_df, "WTG-001")
    """

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        return score_dataframe(df)

    def fleet_summary(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        return aggregate_fleet_risk(scored_df)

    def turbine_summary(
        self,
        scored_df: pd.DataFrame,
        turbine_id: str,
        last_n: int = 360,
    ) -> Dict:
        return turbine_risk_summary(scored_df, turbine_id, last_n)

    @staticmethod
    def score_single(
        anomaly_score: float,
        is_anomaly: bool,
        fault_type: str,
        fault_confidence: float,
        rul_days: float,
        rul_urgency: str,
    ) -> Dict:
        """Score a single reading — useful for real-time API calls."""
        score, tier, dom, bd = compute_risk_score(
            anomaly_score, is_anomaly, fault_type,
            fault_confidence, rul_days, rul_urgency,
        )
        return {
            "risk_score":      round(score, 4),
            "risk_tier":       tier,
            "dominant_signal": dom,
            "breakdown":       asdict(bd),
        }


# ---------------------------------------------------------------------------
# Stand-alone demo (python -m src.risk_scorer)
# ---------------------------------------------------------------------------

def _run_demo():
    log.info("=" * 60)
    log.info("Risk Scorer — Phase 4")
    log.info("=" * 60)

    # ── 1. Load pipeline predictions ──────────────────────────────────────
    pipeline_path = Path("outputs/pipeline_predictions.parquet")
    if not pipeline_path.exists():
        log.error("pipeline_predictions.parquet not found. Run model_pipeline.py first.")
        return

    log.info("Loading pipeline predictions …")
    df = pd.read_parquet(pipeline_path)
    log.info("Loaded %d rows × %d columns", *df.shape)

    # Use predicted values, not ground truth — overwrite directly to avoid duplicates
    if "predicted_rul_days" in df.columns:
        df["rul_days"] = df["predicted_rul_days"]
    if "predicted_fault_type" in df.columns:
        df["fault_type"] = df["predicted_fault_type"]  # overwrites ground truth in-place
    if "is_anomaly_prediction" in df.columns and "is_anomaly" not in df.columns:
        df["is_anomaly"] = df["is_anomaly_prediction"]
    log.info("Mapped predicted columns → scorer expected names")

    # Ensure no duplicate columns
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # Fallback defaults for any still-missing columns
    if "anomaly_score" not in df.columns:
        log.warning("anomaly_score missing — synthesising from is_anomaly flag")
        df["anomaly_score"] = df["is_anomaly"].astype(float) * 0.7 + np.random.uniform(0, 0.3, len(df))
    if "fault_confidence" not in df.columns:
        log.warning("fault_confidence missing — defaulting to 0.5")
        df["fault_confidence"] = 0.5
    if "rul_days" not in df.columns:
        log.warning("rul_days missing — defaulting to 20.0")
        df["rul_days"] = 20.0
    if "rul_urgency" not in df.columns:
        log.warning("rul_urgency missing — defaulting to 'low'")
        df["rul_urgency"] = "low"
    if "fault_type" not in df.columns:
        if "fault_label" in df.columns:
            df["fault_type"] = df["fault_label"]
        else:
            log.warning("fault_type missing — defaulting to 'normal'")
            df["fault_type"] = "normal"
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = False

    # ── 3. Score the DataFrame ─────────────────────────────────────────────
    scorer = RiskScorer()
    scored_df = scorer.score(df)
    log.info("fault_type dtype: %s", scored_df["fault_type"].dtype)
    log.info("fault_type sample: %s", scored_df["fault_type"].head(3).tolist())
    log.info("fault_type type check: %s", scored_df["fault_type"].apply(type).value_counts().to_dict())
    # ── 4. Fleet summary ──────────────────────────────────────────────────
    fleet = scorer.fleet_summary(scored_df)
    log.info("\n%s\n=== Fleet Risk Summary ===\n%s",
             "=" * 60, fleet.to_string(index=False))

    # ── 5. Per-turbine summaries ───────────────────────────────────────────
    log.info("\n%s\n=== Per-Turbine Risk Summaries (latest window) ===", "=" * 60)
    if "turbine_id" in scored_df.columns:
        for tid in scored_df["turbine_id"].unique():
            summary = scorer.turbine_summary(scored_df, tid)
            log.info(
                "[%s] risk=%.3f (%s) | fault=%s (conf=%.2f) | RUL=%.1f days (%s) | dominant=%s",
                summary["turbine_id"],
                summary["latest_risk_score"],
                summary["risk_tier"],
                summary["fault_type"],
                summary["fault_confidence"],
                summary["rul_days"],
                summary["rul_urgency"],
                summary["dominant_signal"],
            )

    # ── 6. Single-reading API demo ─────────────────────────────────────────
    log.info("\n%s\n=== Single-Reading Score Demo ===", "=" * 60)
    scenarios = [
        dict(anomaly_score=0.92, is_anomaly=True,  fault_type="bearing_failure",
             fault_confidence=0.88, rul_days=1.2, rul_urgency="critical",
             label="Imminent bearing failure"),
        dict(anomaly_score=0.55, is_anomaly=True,  fault_type="normal",
             fault_confidence=0.30, rul_days=10.0, rul_urgency="medium",
             label="Uncertain anomaly, medium RUL"),
        dict(anomaly_score=0.10, is_anomaly=False, fault_type="normal",
             fault_confidence=0.95, rul_days=28.0, rul_urgency="low",
             label="Healthy turbine"),
        dict(anomaly_score=0.78, is_anomaly=True,  fault_type="oil_leak",
             fault_confidence=0.62, rul_days=5.5, rul_urgency="high",
             label="Oil leak, high urgency"),
    ]
    for s in scenarios:
        result = RiskScorer.score_single(
            anomaly_score=s["anomaly_score"],
            is_anomaly=s["is_anomaly"],
            fault_type=s["fault_type"],
            fault_confidence=s["fault_confidence"],
            rul_days=s["rul_days"],
            rul_urgency=s["rul_urgency"],
        )
        log.info(
            "%-42s → risk=%.3f (%s) | dominant=%s",
            s["label"], result["risk_score"], result["risk_tier"],
            result["dominant_signal"],
        )

    # ── 7. Risk tier distribution ─────────────────────────────────────────
    log.info("\n%s\n=== Risk Tier Distribution ===", "=" * 60)
    tier_counts = scored_df["risk_tier"].value_counts().reindex(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0
    )
    total = len(scored_df)
    for tier, count in tier_counts.items():
        bar = "█" * int(30 * count / total)
        log.info("  %-10s %6d (%5.1f%%)  %s", tier, count, 100 * count / total, bar)

    # ── 8. Save outputs ───────────────────────────────────────────────────
    out_scored  = OUTPUT_DIR / "risk_scores.parquet"
    out_fleet   = OUTPUT_DIR / "fleet_risk_summary.csv"

    scored_df.to_parquet(out_scored, index=False)
    fleet.to_csv(out_fleet, index=False)

    log.info("=" * 60)
    log.info("Saved: %s", out_scored)
    log.info("Saved: %s", out_fleet)
    log.info("=" * 60)
    log.info("Phase 4 — Risk Scoring complete")
    log.info("=" * 60)


if __name__ == "__main__":
    _run_demo()