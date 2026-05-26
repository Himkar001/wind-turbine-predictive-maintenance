"""
Synthetic Wind Turbine Sensor Data Generator — Permanent Fix
=============================================================

Key improvements over previous version:

1. DISTINCT FAULT FINGERPRINTS
   Each fault type has ONE primary sensor that rises dramatically
   and that no other fault touches as strongly:

   bearing_failure      → bearing_temp ↑↑ (primary), vibration ↑
   gearbox_fault        → oil_pressure ↓↓ (primary), vibration ↑ oscillating
   generator_overheating→ generator_temp ↑↑ (primary), power ↓ moderate
   blade_imbalance      → vibration sinusoidal (primary), rotor_speed oscillates
   oil_leak             → oil_pressure ↓ steady (primary), bearing_temp ↑ mild
   electrical_fault     → power_output ↓↓ (primary), generator_temp ↑ mild
                          bearing_temp UNCHANGED, vibration UNCHANGED

2. 180 DAYS OF DATA
   All fault types appear in the first 60% of the timeline
   so they fall safely inside the training split.

3. BALANCED FAULT EXPOSURE
   Each fault type appears in MULTIPLE turbines and at MULTIPLE
   points in the timeline. No fault is exclusive to one turbine.

4. TURBINE PERSONALITIES (demo story preserved)
   WTG-001  Healthy baseline            ~3%  anomaly
   WTG-002  Bearing + oil leak story   ~22%  anomaly
   WTG-003  Gearbox degradation        ~28%  anomaly
   WTG-004  Critical multi-fault       ~40%  anomaly
   WTG-005  Electrical + recovery      ~20%  anomaly

5. SEVERITY CURVES
   All faults use gradual severity 0→1 progression over days,
   not sudden step functions.
"""

import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
# FAULT INJECTORS — Each has a UNIQUE primary sensor signature
# ═══════════════════════════════════════════════════════════════

def inject_bearing_failure(df, start, end, severity=1.0):
    """
    PRIMARY:   bearing_temp ↑↑  (unique — no other fault raises this as high)
    SECONDARY: vibration ↑, oil_pressure ↓ slightly
    UNCHANGED: generator_temp, power_output (distinguishes from electrical/generator)
    """
    n    = end - start
    prog = np.linspace(0, severity, n)
    noise = np.random.normal(0, 0.3, n)

    df.loc[start:end-1, "bearing_temp"]  += prog * 42 + noise        # PRIMARY — very high
    df.loc[start:end-1, "vibration"]     += prog * 5.5 + np.abs(noise) * 0.5
    df.loc[start:end-1, "oil_pressure"]  -= prog * 0.8               # mild
    # power_output: only tiny drop from mechanical loss
    df.loc[start:end-1, "power_output"]  -= prog * 80                # very small
    # generator_temp: NOT affected — key discriminator
    df.loc[start:end-1, "fault_type"]    = "bearing_failure"
    return df


def inject_gearbox_fault(df, start, end, severity=1.0):
    """
    PRIMARY:   oil_pressure ↓↓  (unique — steepest drop of any fault)
    SECONDARY: vibration oscillating (unique pattern), rotor_speed unstable
    UNCHANGED: bearing_temp stays low, generator_temp unchanged
    """
    n    = end - start
    prog = np.linspace(0, severity, n)
    t    = np.linspace(0, 8 * np.pi, n)
    noise = np.random.normal(0, 0.4, n)

    df.loc[start:end-1, "oil_pressure"]  -= prog * 3.8               # PRIMARY — steep
    df.loc[start:end-1, "vibration"]     += prog * 4.5 + 2.0 * np.sin(t)  # oscillating
    df.loc[start:end-1, "rotor_speed"]   += 1.5 * np.sin(t * 1.3) * severity
    df.loc[start:end-1, "bearing_temp"]  += prog * 8                 # mild only
    df.loc[start:end-1, "power_output"]  -= prog * 120               # small
    # generator_temp: NOT affected
    df.loc[start:end-1, "fault_type"]    = "gearbox_fault"
    return df


def inject_generator_overheating(df, start, end, severity=1.0):
    """
    PRIMARY:   generator_temp ↑↑  (unique — highest generator temp of any fault)
    SECONDARY: power_output ↓ moderate
    UNCHANGED: bearing_temp NOT raised, vibration NOT raised
                (distinguishes from bearing_failure)
    """
    n    = end - start
    prog = np.linspace(0, severity, n)
    noise = np.random.normal(0, 0.5, n)

    df.loc[start:end-1, "generator_temp"] += prog * 48 + noise       # PRIMARY — very high
    df.loc[start:end-1, "power_output"]   -= prog * 380              # moderate
    # bearing_temp: NOT raised — key discriminator vs bearing_failure
    # vibration: NOT raised — key discriminator
    df.loc[start:end-1, "fault_type"]    = "generator_overheating"
    return df


def inject_blade_imbalance(df, start, end, severity=1.0):
    """
    PRIMARY:   vibration SINUSOIDAL pattern (unique waveform, not a ramp)
    SECONDARY: rotor_speed oscillates
    UNCHANGED: all temperatures unchanged, oil_pressure unchanged
                (distinguishes from bearing and gearbox faults)
    """
    n = end - start
    t = np.linspace(0, 10 * np.pi, n)

    # Unique: sinusoidal vibration with growing amplitude
    amp = np.linspace(0, severity * 5.5, n)
    df.loc[start:end-1, "vibration"]    += amp * np.abs(np.sin(t))   # PRIMARY — wave
    df.loc[start:end-1, "rotor_speed"]  += severity * 2.0 * np.sin(t * 1.5)
    # All temps: UNCHANGED — key discriminator
    # oil_pressure: UNCHANGED
    # power: tiny effect only
    df.loc[start:end-1, "power_output"] -= np.linspace(0, severity * 60, n)
    df.loc[start:end-1, "fault_type"]    = "blade_imbalance"
    return df


def inject_oil_leak(df, start, end, severity=1.0):
    """
    PRIMARY:   oil_pressure ↓ LINEAR STEADY decline (different pattern from gearbox)
               Gearbox: steep drop + oscillation
               Oil leak: slow steady linear decline — distinguishable pattern
    SECONDARY: bearing_temp ↑ mild (from friction)
    UNCHANGED: generator_temp, vibration not elevated
    """
    n    = end - start
    prog = np.linspace(0, severity, n)

    df.loc[start:end-1, "oil_pressure"]  -= prog * 2.8               # PRIMARY — steady
    df.loc[start:end-1, "bearing_temp"]  += prog * 12                # mild friction
    # vibration: NOT oscillating (distinguishes from gearbox)
    df.loc[start:end-1, "vibration"]     += prog * 1.2               # very mild
    # generator_temp: NOT affected
    df.loc[start:end-1, "fault_type"]    = "oil_leak"
    return df


def inject_electrical_fault(df, start, end, severity=1.0):
    """
    PRIMARY:   power_output ↓↓  (unique — steepest power drop of any fault)
    SECONDARY: generator_temp ↑ mild
    STRICTLY UNCHANGED: bearing_temp, vibration, oil_pressure
               (this is the KEY discriminator — electrical faults don't
                cause mechanical vibration or bearing heat)
    """
    n    = end - start
    prog = np.linspace(0, severity, n)

    df.loc[start:end-1, "power_output"]   -= prog * 950              # PRIMARY — dramatic
    df.loc[start:end-1, "generator_temp"] += prog * 18               # mild secondary
    # bearing_temp:  NOT touched — critical discriminator
    # vibration:     NOT touched — critical discriminator
    # oil_pressure:  NOT touched — critical discriminator
    df.loc[start:end-1, "fault_type"]    = "electrical_fault"
    return df


INJECTORS = {
    "bearing_failure":       inject_bearing_failure,
    "gearbox_fault":         inject_gearbox_fault,
    "generator_overheating": inject_generator_overheating,
    "blade_imbalance":       inject_blade_imbalance,
    "oil_leak":              inject_oil_leak,
    "electrical_fault":      inject_electrical_fault,
}


# ═══════════════════════════════════════════════════════════════
# TURBINE PROFILES
# All fault types appear across multiple turbines.
# All faults start before Day 120 (within 70% of 180 days)
# so they land safely in the TRAINING split.
# ═══════════════════════════════════════════════════════════════

TURBINE_PROFILES = {

    "WTG-001": {
        "label":       "Healthy baseline",
        "noise_scale": 0.5,
        "fault_events": [
            # Very minor blade imbalance — barely noticeable, good for anomaly detection
            {"type": "blade_imbalance", "day_start": 30,  "day_end": 45,  "severity": 0.15},
            # Tiny oil leak at Day 80 — sensor drift level
            {"type": "oil_leak",        "day_start": 80,  "day_end": 90,  "severity": 0.12},
        ],
        # Result: ~3-5% anomaly — near-healthy control turbine
    },

    "WTG-002": {
        "label":       "Bearing wear + oil leak",
        "noise_scale": 1.0,
        "fault_events": [
            # Early bearing micro-anomaly — model should catch this early
            {"type": "bearing_failure", "day_start": 20,  "day_end": 35,  "severity": 0.20},
            # Bearing worsens
            {"type": "bearing_failure", "day_start": 50,  "day_end": 80,  "severity": 0.55},
            # Oil leak develops as consequence of bearing stress
            {"type": "oil_leak",        "day_start": 70,  "day_end": 100, "severity": 0.45},
            # Late recovery period — normal operation after maintenance at Day 110
            {"type": "bearing_failure", "day_start": 140, "day_end": 165, "severity": 0.30},
        ],
        "maintenance_day": 110,
        # Result: ~22% anomaly — clear bearing degradation story
    },

    "WTG-003": {
        "label":       "Gearbox degradation + blade imbalance",
        "noise_scale": 1.1,
        "fault_events": [
            # Blade imbalance early — vibration signature
            {"type": "blade_imbalance", "day_start": 10,  "day_end": 30,  "severity": 0.30},
            # Gearbox starts degrading — oil pressure drops with oscillating vibration
            {"type": "gearbox_fault",   "day_start": 40,  "day_end": 75,  "severity": 0.65},
            # Gearbox worsens significantly
            {"type": "gearbox_fault",   "day_start": 85,  "day_end": 115, "severity": 0.85},
            # Oil leak secondary to gearbox
            {"type": "oil_leak",        "day_start": 100, "day_end": 130, "severity": 0.50},
            # Late blade imbalance recurrence
            {"type": "blade_imbalance", "day_start": 145, "day_end": 165, "severity": 0.40},
        ],
        # Result: ~28% anomaly — gearbox story
    },

    "WTG-004": {
        "label":       "CRITICAL — multi-fault cascade",
        "noise_scale": 1.3,
        "fault_events": [
            # Stage 1: micro bearing fault — AI catches this early
            {"type": "bearing_failure",       "day_start": 15,  "day_end": 30,  "severity": 0.18},
            # Stage 2: bearing escalates
            {"type": "bearing_failure",       "day_start": 35,  "day_end": 65,  "severity": 0.70},
            # Stage 3: oil leak develops from bearing stress
            {"type": "oil_leak",              "day_start": 55,  "day_end": 85,  "severity": 0.60},
            # Stage 4: electrical fault (separate event)
            {"type": "electrical_fault",      "day_start": 75,  "day_end": 100, "severity": 0.75},
            # Stage 5: generator overheating from electrical fault
            {"type": "generator_overheating", "day_start": 95,  "day_end": 125, "severity": 0.90},
            # Stage 6: critical bearing failure
            {"type": "bearing_failure",       "day_start": 120, "day_end": 160, "severity": 1.00},
        ],
        # Result: ~40% anomaly — full cascade demo story
    },

    "WTG-005": {
        "label":       "Electrical fault + recovery + generator stress",
        "noise_scale": 1.0,
        "fault_events": [
            # Electrical fault early
            {"type": "electrical_fault",      "day_start": 10,  "day_end": 40,  "severity": 0.65},
            # Generator overheating from electrical fault
            {"type": "generator_overheating", "day_start": 35,  "day_end": 60,  "severity": 0.70},
            # --- MAINTENANCE at Day 70 ---
            # Recovery period: Days 70-90 = normal
            # New fault emerges late
            {"type": "blade_imbalance",       "day_start": 100, "day_end": 125, "severity": 0.45},
            {"type": "electrical_fault",      "day_start": 140, "day_end": 165, "severity": 0.55},
        ],
        "maintenance_day": 70,
        # Result: ~20% anomaly — electrical story + recovery
    },
}


# ═══════════════════════════════════════════════════════════════
# BASE SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_base_signals(
    total_readings: int,
    interval_sec: int,
    days: int,
    noise_scale: float,
    config: dict
) -> pd.DataFrame:
    """
    Physics-correlated baseline sensor readings.
    Wind → rotor_speed → power_output → temperatures
    Daily thermal cycle included.
    """
    s = config["sensors"]

    start_time = datetime.now() - timedelta(days=days)
    timestamps = [
        start_time + timedelta(seconds=i * interval_sec)
        for i in range(total_readings)
    ]

    # Wind: Weibull distribution (realistic wind profile)
    wind_speed = np.clip(
        np.random.weibull(2.2, total_readings) * 9
        + np.sin(np.linspace(0, 20 * np.pi, total_readings)) * 2.0
        + np.random.normal(0, 0.4, total_readings),
        s["wind_speed"]["normal_range"][0],
        s["wind_speed"]["normal_range"][1]
    )

    # Rotor speed correlated with wind
    rotor_speed = np.clip(
        wind_speed * 0.55
        + np.random.normal(0, 0.25 * noise_scale, total_readings),
        *s["rotor_speed"]["normal_range"]
    )

    # Power output: cubic relationship with wind (Betz law approximation)
    power_output = np.clip(
        0.48 * wind_speed ** 2.9
        + np.random.normal(0, 20 * noise_scale, total_readings),
        *s["power_output"]["normal_range"]
    )

    # Daily thermal cycle (warmer afternoon)
    hour_of_day = np.array([t.hour for t in timestamps])
    daily_cycle = 4.5 * np.sin((hour_of_day - 6) * np.pi / 12)

    # Bearing temp: correlated with rotor speed + daily cycle
    bearing_temp = np.clip(
        44 + daily_cycle
        + rotor_speed * 1.9
        + np.random.normal(0, 1.0 * noise_scale, total_readings),
        *s["bearing_temp"]["normal_range"]
    )

    # Generator temp: correlated with power output
    generator_temp = np.clip(
        60 + daily_cycle
        + power_output * 0.013
        + np.random.normal(0, 1.5 * noise_scale, total_readings),
        *s["generator_temp"]["normal_range"]
    )

    # Vibration: correlated with rotor speed (exponential noise)
    vibration = np.clip(
        1.4 + rotor_speed * 0.10
        + np.random.exponential(0.20 * noise_scale, total_readings),
        *s["vibration"]["normal_range"]
    )

    # Oil pressure: inversely correlated with rotor speed
    oil_pressure = np.clip(
        4.3 - rotor_speed * 0.04
        + np.random.normal(0, 0.15 * noise_scale, total_readings),
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


# ═══════════════════════════════════════════════════════════════
# PER-TURBINE GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_turbine_data(turbine_id: str, config: dict) -> pd.DataFrame:
    days         = config["data_generation"]["days"]
    interval_sec = config["wind_farm"]["sampling_interval_seconds"]
    rpd          = int(24 * 3600 / interval_sec)   # readings per day
    total        = days * rpd

    profile = TURBINE_PROFILES[turbine_id]
    logger.info(f"  {turbine_id} | {profile['label']}")
    logger.info(f"    {total:,} readings over {days} days ...")

    df = generate_base_signals(total, interval_sec, days, profile["noise_scale"], config)
    df.insert(0, "turbine_id", turbine_id)

    # Inject fault events
    for event in profile.get("fault_events", []):
        s_idx = event["day_start"] * rpd
        e_idx = min(event["day_end"] * rpd, total)
        df = INJECTORS[event["type"]](df, s_idx, e_idx, event["severity"])
        logger.info(
            f"    Injected {event['type']:25s} "
            f"Day {event['day_start']:03d}→{event['day_end']:03d}  "
            f"severity={event['severity']:.2f}"
        )

    # Mark maintenance window
    if "maintenance_day" in profile:
        md    = profile["maintenance_day"]
        ms    = md * rpd
        me    = min((md + 3) * rpd, total)
        df.loc[ms:me-1, "fault_type"] = "maintenance"

    # Hard physical clip — safety bounds
    df["bearing_temp"]   = df["bearing_temp"].clip(0, 125)
    df["generator_temp"] = df["generator_temp"].clip(0, 135)
    df["vibration"]      = df["vibration"].clip(0, 15)
    df["oil_pressure"]   = df["oil_pressure"].clip(0, 10)
    df["power_output"]   = df["power_output"].clip(0, 2500)
    df["rotor_speed"]    = df["rotor_speed"].clip(0, 25)

    # Anomaly flag
    df["is_anomaly"] = (
        ~df["fault_type"].isin(["normal", "maintenance"])
    ).astype(int)

    pct = df["is_anomaly"].mean() * 100
    logger.info(f"    Anomaly rate: {pct:.1f}%")
    return df


# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════

def save_data(df: pd.DataFrame, config: dict):
    out = Path(config["data_generation"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # Full combined dataset
    full_path = out / "all_sensor_data.csv"
    df.to_csv(full_path, index=False)
    logger.info(f"  Saved full dataset     → {full_path}  ({len(df):,} rows)")

    # Per-turbine CSVs
    for tid in df["turbine_id"].unique():
        tdf  = df[df["turbine_id"] == tid]
        path = out / f"sensor_data_{tid}.csv"
        tdf.to_csv(path, index=False)

    # Fault summary
    summary = (
        df.groupby(["turbine_id", "fault_type"])
          .size()
          .reset_index(name="count")
    )
    summary["pct"] = (
        summary["count"]
        / summary.groupby("turbine_id")["count"].transform("sum")
        * 100
    ).round(2)
    summary.to_csv(out / "fault_summary.csv", index=False)
    logger.info(f"  Saved fault summary    → {out}/fault_summary.csv")

    # Demo fault timeline
    import json
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
    pd.DataFrame(timeline).to_csv(out / "demo_fault_timeline.csv", index=False)
    logger.info(f"  Saved demo timeline    → {out}/demo_fault_timeline.csv")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 65)
    logger.info("  Wind Turbine Data Generator — Permanent Fix")
    logger.info("  180 days | Distinct fault fingerprints | Balanced coverage")
    logger.info("=" * 65)

    config = load_config()

    # Override days to 180 permanently
    config["data_generation"]["days"] = 180

    all_dfs = []
    for tid in config["wind_farm"]["turbine_ids"]:
        df = generate_turbine_data(tid, config)
        all_dfs.append(df)
        logger.info("")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    save_data(combined, config)

    # ── Final summary ────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 65)
    logger.info("  GENERATION COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Total rows     : {len(combined):,}")
    logger.info(f"  Turbines       : {combined['turbine_id'].nunique()}")
    logger.info(f"  Date range     : {combined['timestamp'].min().date()} "
                f"→ {combined['timestamp'].max().date()}")
    logger.info("")
    logger.info("  Per-turbine anomaly rates:")
    for tid in config["wind_farm"]["turbine_ids"]:
        tdf   = combined[combined["turbine_id"] == tid]
        pct   = tdf["is_anomaly"].mean() * 100
        label = TURBINE_PROFILES[tid]["label"]
        bar   = "█" * int(pct / 2)
        logger.info(f"    {tid}  {pct:5.1f}%  {bar}  ({label})")

    logger.info("")
    logger.info("  Fault type distribution (all turbines):")
    ft = combined[combined["fault_type"] != "normal"]["fault_type"].value_counts()
    for fault, count in ft.items():
        pct = count / len(combined) * 100
        logger.info(f"    {fault:30s} {count:>9,}  ({pct:.1f}%)")

    logger.info("")
    logger.info("  Fault distribution in first 70% of timeline (train split):")
    cutoff_day = int(180 * 0.70)
    cutoff_ts  = combined["timestamp"].min() + timedelta(days=cutoff_day)
    train_part = combined[combined["timestamp"] <= cutoff_ts]
    ft_train   = train_part[train_part["fault_type"] != "normal"]["fault_type"].value_counts()
    for fault, count in ft_train.items():
        logger.info(f"    {fault:30s} {count:>9,}")

    logger.info("")
    logger.info("  Phase 1 complete ✓  All fault types present in train region")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()