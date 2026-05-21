"""
Synthetic Wind Turbine Sensor Data Generator — Phase 1 (Demo Mode)
===================================================================
Each turbine has a unique personality and fault timeline, designed
to tell a compelling story during project demos:

  WTG-001  Healthy baseline        ~2%  anomaly  (control turbine)
  WTG-002  Minor bearing wear      ~8%  anomaly  (early warning case)
  WTG-003  Gearbox degrading       ~20% anomaly  (intervention needed)
  WTG-004  Critical bearing failure ~35% anomaly  (imminent failure)
  WTG-005  Post-maintenance recovery ~5% → ~25%  (maintenance story)
"""

import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Fault Injectors ───────────────────────────────────────────────────────────

def inject_bearing_failure(df, start, end, severity=1.0):
    """
    Gradual bearing degradation: temp and vibration rise together.
    severity 0.0→1.0 controls how bad the fault gets.
    """
    n = end - start
    prog = np.linspace(0, severity, n)
    noise = np.random.normal(0, 0.4, n)
    df.loc[start:end-1, "bearing_temp"]  += prog * 38 + noise
    df.loc[start:end-1, "vibration"]     += prog * 7  + np.abs(noise)
    df.loc[start:end-1, "oil_pressure"]  -= prog * 1.2
    df.loc[start:end-1, "fault_type"]     = "bearing_failure"
    return df


def inject_gearbox_fault(df, start, end, severity=1.0):
    """
    Gearbox fault: vibration spikes, oil pressure drops, rotor unstable.
    """
    n = end - start
    prog  = np.linspace(0, severity, n)
    noise = np.random.normal(0, 0.6, n)
    df.loc[start:end-1, "vibration"]    += prog * 6.5 + np.abs(noise)
    df.loc[start:end-1, "oil_pressure"] -= prog * 3.0
    df.loc[start:end-1, "rotor_speed"]  += noise * 1.8
    df.loc[start:end-1, "bearing_temp"] += prog * 12
    df.loc[start:end-1, "fault_type"]    = "gearbox_fault"
    return df


def inject_generator_overheating(df, start, end, severity=1.0):
    """
    Generator overheating: temp spikes, power output drops.
    """
    n = end - start
    prog = np.linspace(0, severity, n)
    df.loc[start:end-1, "generator_temp"] += prog * 42
    df.loc[start:end-1, "power_output"]   -= prog * 600
    df.loc[start:end-1, "fault_type"]      = "generator_overheating"
    return df


def inject_blade_imbalance(df, start, end, severity=1.0):
    """
    Blade imbalance: sinusoidal vibration + rotor speed oscillation.
    """
    n = end - start
    t = np.linspace(0, 6 * np.pi, n)
    df.loc[start:end-1, "vibration"]   += severity * (4 + 3 * np.sin(t))
    df.loc[start:end-1, "rotor_speed"] += severity * 2.5 * np.sin(t * 1.5)
    df.loc[start:end-1, "fault_type"]   = "blade_imbalance"
    return df


def inject_oil_leak(df, start, end, severity=1.0):
    """
    Oil leak: steady pressure decline + bearing temp rise from friction.
    """
    n = end - start
    prog = np.linspace(0, severity, n)
    df.loc[start:end-1, "oil_pressure"] -= prog * 3.8
    df.loc[start:end-1, "bearing_temp"] += prog * 18
    df.loc[start:end-1, "vibration"]    += prog * 2.5
    df.loc[start:end-1, "fault_type"]    = "oil_leak"
    return df


def inject_electrical_fault(df, start, end, severity=1.0):
    """
    Electrical fault: sudden power drop + generator temp rise.
    """
    n = end - start
    prog = np.linspace(0, severity, n)
    df.loc[start:end-1, "power_output"]   -= prog * 900
    df.loc[start:end-1, "generator_temp"] += prog * 22
    df.loc[start:end-1, "fault_type"]      = "electrical_fault"
    return df


INJECTORS = {
    "bearing_failure":       inject_bearing_failure,
    "gearbox_fault":         inject_gearbox_fault,
    "generator_overheating": inject_generator_overheating,
    "blade_imbalance":       inject_blade_imbalance,
    "oil_leak":              inject_oil_leak,
    "electrical_fault":      inject_electrical_fault,
}


# ── Turbine Personalities ─────────────────────────────────────────────────────

TURBINE_PROFILES = {

    "WTG-001": {
        "label":       "Healthy baseline",
        "noise_scale": 0.6,      # very low sensor noise
        "fault_events": [],      # no faults — pure healthy data
    },

    "WTG-002": {
        "label":       "Minor bearing wear",
        "noise_scale": 1.0,
        "fault_events": [
            # One slow-developing bearing fault around Day 50, mild severity
            {"type": "bearing_failure", "day_start": 50, "day_end": 75, "severity": 0.45},
            # Small oil leak late in timeline
            {"type": "oil_leak",        "day_start": 78, "day_end": 88, "severity": 0.30},
        ],
    },

    "WTG-003": {
        "label":       "Gearbox degrading — needs attention",
        "noise_scale": 1.2,
        "fault_events": [
            # Early micro-anomalies — barely noticeable
            {"type": "blade_imbalance", "day_start": 10, "day_end": 22, "severity": 0.25},
            # Gearbox starts degrading from Day 30, escalating
            {"type": "gearbox_fault",   "day_start": 30, "day_end": 55, "severity": 0.60},
            # Secondary bearing stress from gearbox load
            {"type": "bearing_failure", "day_start": 50, "day_end": 70, "severity": 0.50},
            # Electrical follow-on fault
            {"type": "electrical_fault","day_start": 68, "day_end": 80, "severity": 0.40},
        ],
    },

    "WTG-004": {
        "label":       "CRITICAL — bearing failure in progress",
        "noise_scale": 1.5,
        "fault_events": [
            # Early micro-anomaly — model should catch this
            {"type": "bearing_failure", "day_start": 15, "day_end": 25, "severity": 0.20},
            # Escalation phase
            {"type": "bearing_failure", "day_start": 25, "day_end": 50, "severity": 0.65},
            # Oil leak develops as consequence
            {"type": "oil_leak",        "day_start": 40, "day_end": 60, "severity": 0.55},
            # Full critical failure
            {"type": "bearing_failure", "day_start": 60, "day_end": 85, "severity": 1.00},
            # Generator overheats from mechanical failure
            {"type": "generator_overheating","day_start": 75,"day_end": 89,"severity": 0.90},
        ],
    },

    "WTG-005": {
        "label":       "Post-maintenance recovery story",
        "noise_scale": 1.1,
        "fault_events": [
            # Pre-maintenance: bad gearbox fault
            {"type": "gearbox_fault",   "day_start":  5, "day_end": 35, "severity": 0.75},
            {"type": "blade_imbalance", "day_start": 20, "day_end": 38, "severity": 0.60},
            # --- MAINTENANCE PERFORMED around Day 40 ---
            # Post-maintenance: clean data for 20 days (recovery window)
            # Then a new minor fault develops toward the end
            {"type": "bearing_failure", "day_start": 68, "day_end": 82, "severity": 0.35},
        ],
        "maintenance_day": 40,   # used for annotation
    },
}


# ── Base Signal Generator ─────────────────────────────────────────────────────

def generate_base_signals(total_readings: int, interval_sec: int,
                          days: int, noise_scale: float,
                          config: dict) -> pd.DataFrame:
    """
    Generate physics-correlated baseline sensor readings.
    Wind speed drives rotor speed → power output → temperatures.
    """
    s = config["sensors"]

    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(seconds=i * interval_sec)
                  for i in range(total_readings)]

    # Wind: Weibull distribution + slow sinusoidal trend (day/night)
    wind_speed = np.clip(
        np.random.weibull(2.2, total_readings) * 9
        + np.sin(np.linspace(0, 14 * np.pi, total_readings)) * 2.5
        + np.random.normal(0, 0.5, total_readings),
        s["wind_speed"]["normal_range"][0],
        s["wind_speed"]["normal_range"][1]
    )

    rotor_speed = np.clip(
        wind_speed * 0.55 + np.random.normal(0, 0.3 * noise_scale, total_readings),
        *s["rotor_speed"]["normal_range"]
    )

    power_output = np.clip(
        0.5 * wind_speed ** 2.8 + np.random.normal(0, 25 * noise_scale, total_readings),
        *s["power_output"]["normal_range"]
    )

    hour_of_day = np.array([t.hour for t in timestamps])
    daily_cycle = 5 * np.sin((hour_of_day - 6) * np.pi / 12)

    bearing_temp = np.clip(
        45 + daily_cycle + rotor_speed * 1.8
        + np.random.normal(0, 1.2 * noise_scale, total_readings),
        *s["bearing_temp"]["normal_range"]
    )

    generator_temp = np.clip(
        62 + daily_cycle + power_output * 0.014
        + np.random.normal(0, 1.8 * noise_scale, total_readings),
        *s["generator_temp"]["normal_range"]
    )

    vibration = np.clip(
        1.5 + rotor_speed * 0.12
        + np.random.exponential(0.25 * noise_scale, total_readings),
        *s["vibration"]["normal_range"]
    )

    oil_pressure = np.clip(
        4.2 - rotor_speed * 0.05
        + np.random.normal(0, 0.18 * noise_scale, total_readings),
        *s["oil_pressure"]["normal_range"]
    )

    return pd.DataFrame({
        "timestamp":      timestamps,
        "wind_speed":     np.round(wind_speed, 3),
        "rotor_speed":    np.round(rotor_speed, 3),
        "bearing_temp":   np.round(bearing_temp, 3),
        "generator_temp": np.round(generator_temp, 3),
        "vibration":      np.round(vibration, 3),
        "oil_pressure":   np.round(oil_pressure, 3),
        "power_output":   np.round(power_output, 3),
        "fault_type":     "normal",
        "is_anomaly":     0,
        "severity":       0.0,
    })


# ── Per-Turbine Generator ─────────────────────────────────────────────────────

def generate_turbine_data(turbine_id: str, config: dict) -> pd.DataFrame:
    days         = config["data_generation"]["days"]
    interval_sec = config["wind_farm"]["sampling_interval_seconds"]
    readings_per_day = int(24 * 3600 / interval_sec)
    total_readings   = days * readings_per_day

    profile = TURBINE_PROFILES[turbine_id]
    logger.info(f"  {turbine_id} | {profile['label']}")
    logger.info(f"    Generating {total_readings:,} readings over {days} days ...")

    df = generate_base_signals(
        total_readings, interval_sec, days,
        profile["noise_scale"], config
    )
    df.insert(0, "turbine_id", turbine_id)

    # ── Inject fault events ──────────────────────
    for event in profile.get("fault_events", []):
        start_idx = event["day_start"] * readings_per_day
        end_idx   = min(event["day_end"] * readings_per_day, total_readings)
        injector  = INJECTORS[event["type"]]
        df = injector(df, start_idx, end_idx, event["severity"])
        logger.info(
            f"    Injected {event['type']:25s} "
            f"Day {event['day_start']:02d}→{event['day_end']:02d}  "
            f"severity={event['severity']:.2f}"
        )

    # ── Mark maintenance window for WTG-005 ─────
    if "maintenance_day" in profile:
        mday = profile["maintenance_day"]
        m_start = mday * readings_per_day
        m_end   = min((mday + 3) * readings_per_day, total_readings)
        df.loc[m_start:m_end-1, "fault_type"] = "maintenance"

    # ── Physical bounds (safety clip) ───────────
    df["bearing_temp"]   = df["bearing_temp"].clip(0, 120)
    df["generator_temp"] = df["generator_temp"].clip(0, 135)
    df["vibration"]      = df["vibration"].clip(0, 15)
    df["oil_pressure"]   = df["oil_pressure"].clip(0, 10)
    df["power_output"]   = df["power_output"].clip(0, 2500)
    df["rotor_speed"]    = df["rotor_speed"].clip(0, 25)

    # ── Anomaly flag ─────────────────────────────
    df["is_anomaly"] = (
        ~df["fault_type"].isin(["normal", "maintenance"])
    ).astype(int)

    anomaly_pct = df["is_anomaly"].mean() * 100
    logger.info(f"    Anomaly rate: {anomaly_pct:.1f}%  ✓")
    return df


# ── Save ──────────────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, config: dict):
    output_dir = Path(config["data_generation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full combined dataset
    full_path = output_dir / "sensor_data_all.csv"
    df.to_csv(full_path, index=False)
    logger.info(f"  Saved full dataset     → {full_path}  ({len(df):,} rows)")

    # Per-turbine CSVs
    for tid in df["turbine_id"].unique():
        tdf  = df[df["turbine_id"] == tid]
        path = output_dir / f"sensor_data_{tid}.csv"
        tdf.to_csv(path, index=False)

    # Fault summary
    summary = (
        df.groupby(["turbine_id", "fault_type"])
          .size()
          .reset_index(name="count")
    )
    summary["percentage"] = (
        summary["count"] / summary.groupby("turbine_id")["count"]
        .transform("sum") * 100
    ).round(2)
    summary_path = output_dir / "fault_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"  Saved fault summary    → {summary_path}")

    # Demo timeline (useful for dashboard annotations)
    timeline = []
    for tid, profile in TURBINE_PROFILES.items():
        for event in profile.get("fault_events", []):
            timeline.append({
                "turbine_id":  tid,
                "fault_type":  event["type"],
                "day_start":   event["day_start"],
                "day_end":     event["day_end"],
                "severity":    event["severity"],
                "label":       profile["label"],
            })
    timeline_path = output_dir / "demo_fault_timeline.csv"
    pd.DataFrame(timeline).to_csv(timeline_path, index=False)
    logger.info(f"  Saved demo timeline    → {timeline_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  Wind Turbine Data Generator — Phase 1 (Demo Mode)")
    logger.info("=" * 60)

    config = load_config()
    all_dfs = []

    for tid in config["wind_farm"]["turbine_ids"]:
        df = generate_turbine_data(tid, config)
        all_dfs.append(df)
        logger.info("")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    save_data(combined, config)

    # ── Final summary ────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total rows     : {len(combined):,}")
    logger.info(f"  Turbines       : {combined['turbine_id'].nunique()}")
    logger.info(f"  Date range     : {combined['timestamp'].min().date()} "
                f"→ {combined['timestamp'].max().date()}")
    logger.info("")
    logger.info("  Per-turbine anomaly rates:")

    for tid in config["wind_farm"]["turbine_ids"]:
        tdf  = combined[combined["turbine_id"] == tid]
        pct  = tdf["is_anomaly"].mean() * 100
        label = TURBINE_PROFILES[tid]["label"]
        bar  = "█" * int(pct / 2)
        logger.info(f"    {tid}  {pct:5.1f}%  {bar}  ({label})")

    logger.info("")
    logger.info("  Fault type distribution:")
    ft = combined[combined["fault_type"] != "normal"]["fault_type"].value_counts()
    for fault, count in ft.items():
        logger.info(f"    {fault:30s} {count:>8,}")

    logger.info("")
    logger.info("  Phase 1 complete ✓  Ready for Phase 2")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
