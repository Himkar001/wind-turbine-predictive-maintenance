# Data Directory

This folder is **git-ignored**. Raw data files are not committed to version control as they are large (400MB+) and fully reproducible from code.

## Regenerate Data

```bash
# From project root
python src/data/generate_data.py
```

Output will appear in `data/raw/`.

---

## Generated Files

| File | Size (approx) | Description |
|---|---|---|
| `all_sensor_data.csv` | ~422 MB | Full combined dataset — all 5 turbines, 90 days |
| `sensor_data_WTG-001.csv` | ~68 MB | Per-turbine file — healthy baseline |
| `sensor_data_WTG-002.csv` | ~80 MB | Per-turbine file — minor bearing wear |
| `sensor_data_WTG-003.csv` | ~91 MB | Per-turbine file — gearbox degrading |
| `sensor_data_WTG-004.csv` | ~95 MB | Per-turbine file — critical failure cascade |
| `sensor_data_WTG-005.csv` | ~88 MB | Per-turbine file — post-maintenance recovery |
| `fault_summary.csv` | ~1 KB | Fault type counts per turbine |
| `demo_fault_timeline.csv` | ~1 KB | Fault event timeline for dashboard annotations |

---

## Dataset Schema

### `all_sensor_data.csv` — 12 columns, ~3.9M rows

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | datetime | — | Reading timestamp (10-second intervals) |
| `turbine_id` | string | — | Turbine identifier (WTG-001 to WTG-005) |
| `wind_speed` | float | m/s | Ambient wind speed |
| `rotor_speed` | float | rpm | Rotor rotation speed |
| `bearing_temp` | float | °C | Main bearing temperature |
| `generator_temp` | float | °C | Generator winding temperature |
| `vibration` | float | mm/s | Vibration amplitude (RMS) |
| `oil_pressure` | float | bar | Gearbox oil pressure |
| `power_output` | float | kW | Active power output |
| `fault_type` | string | — | Fault label (normal / bearing_failure / gearbox_fault / ...) |
| `is_anomaly` | int | 0/1 | Binary anomaly flag |
| `severity` | float | 0.0–1.0 | Fault severity (0 = healthy, 1 = critical) |

---

## Sample Rows

### Normal operation (WTG-001)

```
timestamp            turbine_id  wind_speed  rotor_speed  bearing_temp  generator_temp  vibration  oil_pressure  power_output  fault_type  is_anomaly  severity
2026-02-20 00:00:00  WTG-001     8.412       4.631        52.341        68.214          1.823      4.102         312.540       normal      0           0.0
2026-02-20 00:00:10  WTG-001     8.531       4.694        52.187        68.445          1.791      4.118         318.920       normal      0           0.0
2026-02-20 00:00:20  WTG-001     7.943       4.369        51.876        67.832          1.654      4.089         295.610       normal      0           0.0
2026-02-20 00:00:30  WTG-001     9.102       5.006        53.012        69.103          1.912      4.131         341.870       normal      0           0.0
```

### Early bearing fault developing (WTG-002, Day 50)

```
timestamp            turbine_id  wind_speed  rotor_speed  bearing_temp  generator_temp  vibration  oil_pressure  power_output  fault_type       is_anomaly  severity
2026-04-11 00:00:00  WTG-002     9.231       5.077        61.203        70.412          3.102      3.921         358.210       bearing_failure  1           0.09
2026-04-11 06:00:00  WTG-002     8.874       4.881        64.891        71.023          3.891      3.876         341.560       bearing_failure  1           0.18
2026-04-11 12:00:00  WTG-002     9.541       5.248        68.340        71.841          4.923      3.812         362.490       bearing_failure  1           0.27
2026-04-11 18:00:00  WTG-002     8.102       4.456        72.108        72.310          5.841      3.743         318.920       bearing_failure  1           0.36
```

### Critical failure cascade (WTG-004, Day 75 — bearing + overheating)

```
timestamp            turbine_id  wind_speed  rotor_speed  bearing_temp  generator_temp  vibration  oil_pressure  power_output  fault_type              is_anomaly  severity
2026-05-06 00:00:00  WTG-004     10.231      5.627        98.412        101.230         11.203     1.923         198.340       bearing_failure         1           0.88
2026-05-06 06:00:00  WTG-004     9.871       5.430        102.891       108.412         12.341     1.812         172.560       generator_overheating   1           0.91
2026-05-06 12:00:00  WTG-004     10.540      5.797        107.230       114.023         13.102     1.743         143.210       generator_overheating   1           0.94
2026-05-06 18:00:00  WTG-004     9.102       5.006        111.890       119.341         13.891     1.681         108.430       generator_overheating   1           0.97
```

### Post-maintenance recovery (WTG-005, Day 42 — clean after maintenance)

```
timestamp            turbine_id  wind_speed  rotor_speed  bearing_temp  generator_temp  vibration  oil_pressure  power_output  fault_type  is_anomaly  severity
2026-04-03 00:00:00  WTG-005     8.341       4.588        51.203        67.891          1.743      4.112         305.670       normal      0           0.0
2026-04-03 06:00:00  WTG-005     9.102       5.006        52.341        68.230          1.812      4.098         341.230       normal      0           0.0
2026-04-03 12:00:00  WTG-005     10.231      5.627        53.891        69.412          1.923      4.087         389.560       normal      0           0.0
2026-04-03 18:00:00  WTG-005     7.891       4.340        51.012        67.341          1.654      4.103         289.340       normal      0           0.0
```

---

## Turbine Personalities (Demo Design)

| Turbine | Story | Anomaly Rate | Key Faults |
|---|---|---|---|
| WTG-001 | Healthy baseline — control turbine | ~0% | None |
| WTG-002 | Minor bearing wear developing late | ~39% | bearing_failure, oil_leak |
| WTG-003 | Gearbox degrading — needs attention | ~69% | blade_imbalance, gearbox_fault, bearing_failure, electrical_fault |
| WTG-004 | Critical — multi-fault cascade | ~82% | bearing_failure (3 stages), oil_leak, generator_overheating |
| WTG-005 | Maintenance story — bad → fixed → new fault | ~52% | gearbox_fault, blade_imbalance → maintenance → bearing_failure |

---

## Sensor Normal Operating Ranges

| Sensor | Normal Range | Warning | Critical |
|---|---|---|---|
| Bearing temperature | 20–75 °C | 80 °C | 95 °C |
| Vibration | 0.1–4.5 mm/s | 7.0 mm/s | 10.0 mm/s |
| Rotor speed | 5–15 rpm | 17 rpm | 20 rpm |
| Oil pressure | 2.5–6.0 bar | 2.0 bar | 1.5 bar |
| Generator temperature | 40–90 °C | 95 °C | 110 °C |
| Power output | 100–2000 kW | 50 kW | 20 kW |
| Wind speed | 3–25 m/s | 28 m/s | 30 m/s |

---

## Fault Type Reference

| Fault | Affected Sensors | How It Looks in Data |
|---|---|---|
| `bearing_failure` | bearing_temp ↑, vibration ↑, oil_pressure ↓ | Gradual rise over days |
| `gearbox_fault` | vibration ↑↑, oil_pressure ↓, rotor_speed unstable | Oscillating vibration spike |
| `generator_overheating` | generator_temp ↑↑, power_output ↓ | Rapid temp climb |
| `blade_imbalance` | vibration sinusoidal, rotor_speed oscillates | Regular sine-wave pattern |
| `oil_leak` | oil_pressure ↓ steady, bearing_temp ↑ | Linear pressure decline |
| `electrical_fault` | power_output ↓↓, generator_temp ↑ | Sudden power drop |
| `maintenance` | all sensors normal | Clean readings (WTG-005 Day 40–43) |
