"""
Phase 5 — Turbine Maintenance Knowledge Base
These documents are ingested into ChromaDB and retrieved by the RAG agent.
Each entry maps to a real fault type from the Phase 1 simulator.
"""

MAINTENANCE_DOCS = [
    # ── Bearing Failure ───────────────────────────────────────────────────────
    {
        "id": "kb_bearing_001",
        "title": "Bearing Failure — Diagnosis & Replacement Procedure",
        "fault_type": "bearing_failure",
        "content": """
Bearing failures account for 40-50% of wind turbine drivetrain faults.
Early indicators: elevated bearing temperature (>80°C), increased vibration (>2.5 mm/s RMS),
unusual acoustic emissions, and rising oil contamination levels.

Diagnosis Steps:
1. Confirm bearing temperature trend — a rate of rise >2°C/hour is a critical warning.
2. Perform vibration spectrum analysis — look for defect frequencies (BPFO, BPFI, BSF).
3. Check oil sample for metallic particles (iron content >200 ppm = urgent replacement).
4. Inspect bearing housing for discoloration or oil leakage.

Replacement Procedure:
1. Shut down turbine safely — activate nacelle lock and lockout/tagout.
2. Drain gearbox oil. Remove gearbox access panel.
3. Measure shaft dimensions before removal. Record reference values.
4. Use hydraulic bearing puller — never hammer the bearing race.
5. Clean housing bore; inspect for scoring or corrosion.
6. Heat new bearing to 80–100°C before installation.
7. Refill with fresh oil to specification (ISO VG 320 gear oil typical).
8. Run turbine at 50% load for 4 hours; monitor temperature trend.

Safety Warnings:
- Never exceed 110°C when heating bearing — risk of annealing steel.
- Always support the shaft during removal to prevent drooping.
- Confirm zero-energy state before any mechanical work.

Expected Downtime: 8–24 hours depending on turbine class.
Spare Parts: Main shaft bearing (class-specific), seals, fasteners, lubricant.
        """,
        "source": "IEC 61400-4 Drivetrain Maintenance Manual",
    },

    # ── Blade Imbalance ───────────────────────────────────────────────────────
    {
        "id": "kb_blade_001",
        "title": "Blade Imbalance — Detection & Correction",
        "fault_type": "blade_imbalance",
        "content": """
Blade imbalance generates 1P (once-per-revolution) vibration and increases fatigue loads on
the main shaft, main bearing, and tower. It may result from ice accretion, blade damage,
or pitch system misalignment.

Detection:
1. Elevated 1P vibration in nacelle accelerometer data.
2. Asymmetric power curve — turbine underperforms at certain wind directions.
3. Visual inspection for ice, erosion, cracks, or surface deposits.
4. Blade pitch angle mismatch detected by pitch encoders (>0.3° deviation = investigate).

Correction Procedure:
A) Ice removal: Use de-icing heating elements or wait for ambient clearance.
   Never use mechanical tools on blade surface — risk of delamination.
B) Pitch alignment: Access pitch controller HMI. Recalibrate pitch encoder offsets.
   Perform static pitch balance test at 0° and 90°.
C) Structural repair: If cracks or erosion found, apply composite repair patch.
   Restore leading edge protection tape after repair.
D) Mass balancing: Add or remove balance weights in blade root access panel.
   Target residual imbalance <50 g·m per blade.

Safety Warnings:
- Never access blade root during high winds (>8 m/s at hub).
- Use rated fall-arrest equipment for any elevated work.

Expected Downtime: 2–48 hours depending on severity.
Spare Parts: Leading edge tape, pitch encoder, balance weights.
        """,
        "source": "DNVGL-ST-0376 Blade Maintenance Standard",
    },

    # ── Electrical Fault ──────────────────────────────────────────────────────
    {
        "id": "kb_electrical_001",
        "title": "Electrical Fault — Generator & Converter Troubleshooting",
        "fault_type": "electrical_fault",
        "content": """
Electrical faults in wind turbines typically involve the generator windings, power converter,
transformer, or control electronics. They manifest as abnormal power output, grid fault trips,
and temperature anomalies in the generator.

Common Root Causes:
- Insulation breakdown in generator windings (partial discharge events)
- IGBT module failure in power converter
- Grid voltage unbalance or frequency excursion
- Slip ring and brush wear (DFIG configurations)

Diagnosis Steps:
1. Check SCADA fault log for grid fault codes (over/under voltage, frequency, imbalance).
2. Measure insulation resistance of generator windings — must be >100 MΩ at 500V DC.
3. Inspect converter cabinet: check for blown fuses, tripped breakers, burnt components.
4. For DFIG: inspect slip rings and brushes. Carbon brush wear limit is 25% of new length.
5. Thermal imaging of generator under load to identify hotspots.

Repair Actions:
- Minor: Reset converter, clear fault codes, restart turbine.
- Moderate: Replace blown fuses or IGBT module in converter.
- Major: Generator rewind or replacement — requires specialist contractor.

Safety Warnings:
- LETHAL VOLTAGE — confirm full electrical isolation before entering converter cabinet.
- Capacitors in converter retain charge — wait 5 minutes after isolation and verify with meter.
- Only qualified electricians may perform generator or converter work.

Expected Downtime: 1 hour (reset) to 2 weeks (generator rewind).
Spare Parts: IGBT modules, fuses, slip ring brushes, insulation tape.
        """,
        "source": "IEC 60034-1 Generator Maintenance Guide",
    },

    # ── Gearbox Fault ─────────────────────────────────────────────────────────
    {
        "id": "kb_gearbox_001",
        "title": "Gearbox Fault — Inspection & Oil Management",
        "fault_type": "gearbox_fault",
        "content": """
Gearbox faults are the most costly turbine failures, with replacement costs of $250k–$500k.
Early detection through oil analysis and vibration monitoring is critical to prevent full failure.

Key Indicators:
- Oil temperature >85°C sustained
- Chip detector activation (magnetic plug collects metallic debris)
- Vibration at gear mesh frequencies (GMF = shaft RPM × tooth count)
- Oil pressure drop below minimum specification
- Increased noise signature in nacelle acoustic monitoring

Inspection Procedure:
1. Collect oil sample — test for viscosity, water content, iron, chromium, silicon particles.
2. Remove and inspect magnetic chip detector — quantify and characterise debris.
3. Borescope inspection of gear teeth and planet carrier through inspection ports.
4. Check oil filter differential pressure — replace at ΔP >2.5 bar.
5. Inspect oil cooler for blockage or leakage.

Oil Change / Flush Procedure:
1. Warm gearbox to operating temperature (60–70°C) before drain.
2. Drain completely. Flush with flushing oil; circulate for 30 minutes.
3. Refill to correct level. Use OEM-specified gear oil grade.
4. Install new oil filter. Reset oil life counter.
5. Run at 25% load for 1 hour. Re-sample oil at 50 hours.

Safety Warnings:
- Hot oil drain — wear heat-resistant gloves and face shield.
- Dispose of waste oil per local environmental regulations.
- Never run gearbox with oil below minimum level — catastrophic failure risk.

Expected Downtime: 4 hours (oil change) to 6+ weeks (gearbox replacement).
Spare Parts: Oil filter, magnetic chip detector, gearbox oil, seals.
        """,
        "source": "Vestas/Siemens Gearbox Maintenance Manual Rev 4",
    },

    # ── Generator Overheating ─────────────────────────────────────────────────
    {
        "id": "kb_gen_overheat_001",
        "title": "Generator Overheating — Cooling System Diagnosis",
        "fault_type": "generator_overheating",
        "content": """
Generator overheating reduces insulation life exponentially — every 10°C above rated temperature
halves winding insulation life. Sustained overheating leads to insulation failure and rewinding.

Threshold Limits:
- Winding temperature: Warning >130°C, Shutdown >150°C (Class F insulation typical)
- Bearing temperature: Warning >90°C, Shutdown >110°C
- Cooling air outlet: Warning >60°C

Root Causes:
1. Cooling system blockage — air filter clogged, ventilation ducts obstructed
2. Generator overload — wind resource exceeding design parameters
3. Cooling fan failure — belt slip or motor failure
4. High ambient temperature combined with above-rated operation
5. Partial winding short-circuit increasing I²R losses

Diagnosis:
1. Check cooling air filter — replace if pressure drop >50% of new filter spec.
2. Inspect cooling fan rotation and belt tension.
3. Measure generator current in all three phases — imbalance >5% indicates winding problem.
4. Verify derating curve is applied at high ambient temperatures.

Corrective Actions:
1. Clean or replace air filters (scheduled every 3–6 months).
2. Replace cooling fan belt or fan motor if failed.
3. Apply power derating during high ambient periods (>35°C ambient).
4. If winding temperatures remain elevated after cooling fixes — arrange specialist inspection.

Expected Downtime: 2–4 hours (filter/fan) to weeks (winding repair).
Spare Parts: Air filter, cooling fan belt, fan motor, temperature sensors.
        """,
        "source": "IEC 60034-6 Generator Cooling Standard",
    },

    # ── Oil Leak ──────────────────────────────────────────────────────────────
    {
        "id": "kb_oil_leak_001",
        "title": "Oil Leak — Sealing System Inspection & Repair",
        "fault_type": "oil_leak",
        "content": """
Oil leaks in wind turbines are an environmental hazard and indicate seal degradation.
Left unaddressed they lead to component damage from lubricant starvation.

Common Leak Locations:
- Main gearbox input/output shaft seals
- Pitch gearbox seals
- Hydraulic system connections and hoses
- Transformer oil cooler
- Hub lubrication system

Detection:
1. Visual inspection of nacelle floor, bedplate, and tower interior.
2. Monitor oil level trend in SCADA — level drop >0.5 L/day requires investigation.
3. Oil pressure drop on hydraulic system may indicate external leak.
4. Environmental inspection around tower base for ground contamination.

Repair Procedure:
1. Identify leak source — clean area, dry, and observe during operation.
2. For shaft seals: drain system, remove seal housing, install new seal.
   Use correct lip seal orientation — check OEM diagram.
3. For hydraulic hoses: depressurise system (confirm zero pressure) before disconnecting.
   Replace hose with correct rated replacement.
4. Apply thread sealant on pipe fittings (PTFE tape or Loctite 577).
5. Refill system to correct level. Check all levels after first run.

Safety Warnings:
- Hydraulic systems can be pressurised to 250+ bar — always confirm zero pressure.
- Oil spills create slip hazards — contain immediately with absorbent pads.
- Report any ground contamination to environmental compliance immediately.

Expected Downtime: 2–12 hours.
Spare Parts: Shaft seals, hydraulic hoses, O-rings, absorbent kits.
        """,
        "source": "Wind Turbine O&M Best Practices Guide — DNV GL",
    },

    # ── Scheduled Maintenance ─────────────────────────────────────────────────
    {
        "id": "kb_scheduled_001",
        "title": "Scheduled Preventive Maintenance — Semi-Annual Checklist",
        "fault_type": "maintenance",
        "content": """
Preventive maintenance intervals: Semi-annual (every 6 months) or at defined operating hours.

Semi-Annual PM Checklist:
Mechanical:
- [ ] Inspect rotor blades — erosion, cracks, lightning protection
- [ ] Lubricate all grease points: main bearing, pitch bearings, yaw ring
- [ ] Check bolt torques: tower flanges, nacelle, hub to shaft (use torque record sheet)
- [ ] Inspect main shaft and gearbox for oil leaks
- [ ] Check coupling alignment — realign if >0.1 mm offset
- [ ] Inspect brake pads — replace at 50% wear

Electrical:
- [ ] Thermographic scan of electrical connections
- [ ] Test emergency stop function
- [ ] Clean and inspect slip rings and brushes
- [ ] Verify earthing continuity (<1 Ω)
- [ ] Test lightning protection system continuity

Control & Safety:
- [ ] Test all anemometers and wind vanes — compare against met mast
- [ ] Test over-speed protection function
- [ ] Verify pitch system emergency battery backup
- [ ] Download and review fault logs — clear transient faults
- [ ] Update SCADA firmware if update available

Consumables to Replace:
- Air filters (nacelle and converter cabinet)
- Oil filter (gearbox)
- Coolant (if >2 years old)

Expected Duration: 8–16 hours per turbine.
        """,
        "source": "IEC 61400-5 O&M Requirements Standard",
    },
]