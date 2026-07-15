#!/usr/bin/env python3
"""
Generate synthetic automotive maintenance datasets with realistic multi-part associations.

Outputs:
- parts.csv
- repair_orders.csv
- ro_parts_used.csv
- training_multi_label.csv

Usage:
    python generate_automotive_synthetic.py --output_dir ./auto_data --n_ro 5000 --seed 42
"""

import os
from util import paths
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------
AUTOMOTIVE_MODELS = [
    "SEDAN-A", "SEDAN-B", "SUV-X", "SUV-Z", "TRUCK-T1", "TRUCK-T2", "VAN-C", "COUPE-R"
]

ENVIRONMENTS = ["normal", "high_temp", "high_humidity", "dusty", "city_stopgo", "highway_longhaul"]

FAILURE_MAP = {
    "engine_cooling": ["Coolant temperature high", "Radiator efficiency low", "Coolant leak suspected"],
    "hydraulic_brake": ["Brake pressure drop", "Brake pedal spongy", "Hydraulic pressure unstable"],
    "pneumatic_intake": ["Intake actuator delayed", "Air regulation unstable", "Pneumatic valve not switching"],
    "electrical_power": ["Intermittent electrical fault", "Unexpected power cut", "Control unit reset observed"],
    "drivetrain": ["Drive transfer slip", "Torque transfer delay", "Driveline vibration under load"],
    "conveyor_like_accessory": ["Belt slipping", "Accessory drive noise", "Line speed not maintained"],
}

FAILURE_CODE = {
    "engine_cooling": "FC-CLG-11",
    "hydraulic_brake": "FC-HYD-04",
    "pneumatic_intake": "FC-PNE-07",
    "electrical_power": "FC-ELC-22",
    "drivetrain": "FC-DRV-19",
    "conveyor_like_accessory": "FC-ACC-09",
}

# Association templates (weighted)
ASSOCIATIONS = {
    "engine_cooling": [
        (["coolant_pump", "coolant_filter"], 0.35),
        (["radiator_core", "coolant_filter"], 0.30),
        (["fan", "thermostat"], 0.20),
        (["coolant_pump", "fan", "coolant_filter"], 0.15),
    ],
    "hydraulic_brake": [
        (["pressure_valve", "fluid_filter"], 0.35),
        (["hydraulic_hose", "seal_kit"], 0.30),
        (["pump", "fluid_filter"], 0.20),
        (["hydraulic_hose_revB", "seal_kit_plus"], 0.15),
    ],
    "pneumatic_intake": [
        (["solenoid_valve", "air_line"], 0.40),
        (["frl_unit", "air_regulator"], 0.30),
        (["air_line", "frl_unit", "solenoid_valve"], 0.30),
    ],
    "electrical_power": [
        (["fuse", "relay"], 0.45),
        (["relay", "contactor"], 0.25),
        (["sensor_prox", "relay"], 0.15),
        (["wiring_harness", "relay"], 0.15),
    ],
    "drivetrain": [
        (["drive_chain", "idler_roller"], 0.35),
        (["idler_roller", "tensioner"], 0.35),
        (["drive_chain", "tensioner", "chain_lube"], 0.30),
    ],
    "conveyor_like_accessory": [
        (["belt", "idler_roller"], 0.40),
        (["belt", "tensioner"], 0.35),
        (["belt", "idler_roller", "chain_lube"], 0.25),
    ],
}

# -----------------------------
# Parts master (automotive-only)
# -----------------------------
def build_parts_master(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    base_parts = [
        ("coolant_pump", "cooling", "pumps"),
        ("coolant_filter", "cooling", "filters"),
        ("radiator_core", "cooling", "heat_exchange"),
        ("fan", "cooling", "air_movement"),
        ("thermostat", "cooling", "controls"),

        ("fluid_filter", "hydraulic", "filters"),
        ("pressure_valve", "hydraulic", "valves"),
        ("hydraulic_hose", "hydraulic", "hoses"),
        ("hydraulic_hose_revB", "hydraulic", "hoses"),
        ("pump", "hydraulic", "pumps"),
        ("seal_kit", "mechanical", "seals"),
        ("seal_kit_plus", "mechanical", "seals"),

        ("solenoid_valve", "pneumatic", "valves"),
        ("air_line", "pneumatic", "hoses"),
        ("frl_unit", "pneumatic", "conditioning"),
        ("air_regulator", "pneumatic", "regulators"),

        ("fuse", "electrical", "protection"),
        ("relay", "electrical", "switchgear"),
        ("contactor", "electrical", "switchgear"),
        ("sensor_prox", "electrical", "sensors"),
        ("wiring_harness", "electrical", "cabling"),

        ("drive_chain", "drivetrain", "chains"),
        ("idler_roller", "mechanical", "rollers"),
        ("tensioner", "mechanical", "tensioners"),
        ("belt", "mechanical", "belts"),
        ("chain_lube", "consumable", "lubricants"),
        ("bearing_set", "mechanical", "bearings"),
        ("shaft_seal", "mechanical", "seals"),
        ("high_temp_grease", "consumable", "lubricants"),
        ("spindle_motor", "electromechanical", "motors"),
    ]

    oems = ["OmniDrive", "AtlasMotion", "VectorMech", "HydraCore", "ForgeLine"]

    rows = []
    pid = 1001
    for ptype, cat, subcat in base_parts:
        # create multiple SKU variants per type
        n_variants = 2 if ptype in {"seal_kit", "fuse", "relay", "coolant_filter", "drive_chain"} else 1
        for _ in range(n_variants):
            part_id = f"P-{pid}"
            part_name = f"{ptype.upper()}-{random.randint(10,99)}"
            unit_cost = round(max(8.0, np.random.lognormal(mean=4.4, sigma=0.45)), 2)
            lead = int(np.clip(np.random.normal(6, 4), 1, 25))
            comp_models = "|".join(sorted(random.sample(AUTOMOTIVE_MODELS, k=random.randint(3, len(AUTOMOTIVE_MODELS)))))
            usage_bucket = random.choices(["head", "mid", "tail"], weights=[0.7, 0.2, 0.1])[0]
            sampling_weight = {"head": 0.0042, "mid": 0.00148, "tail": 0.00042}[usage_bucket]

            rows.append({
                "part_id": part_id,
                "part_type": ptype,
                "part_name": part_name,
                "category": cat,
                "subcategory": subcat,
                "oem": random.choice(oems),
                "unit_cost": unit_cost,
                "lead_time_days": lead,
                "compatible_models": comp_models,
                "is_active": True,
                "usage_bucket": usage_bucket,
                "sampling_weight": sampling_weight
            })
            pid += 1

    parts = pd.DataFrame(rows).sort_values("part_id").reset_index(drop=True)
    return parts

# -----------------------------
# Synthetic RO + usage generation
# -----------------------------
def weighted_choice(items_with_w):
    items, weights = zip(*items_with_w)
    idx = random.choices(range(len(items)), weights=weights, k=1)[0]
    return items[idx]

def make_timestamp(start_dt, end_dt):
    delta = end_dt - start_dt
    sec = random.randint(0, int(delta.total_seconds()))
    return start_dt + timedelta(seconds=sec)

def build_datasets(n_ro=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    parts = build_parts_master(seed=seed)
    part_by_type = {t: df["part_id"].tolist() for t, df in parts.groupby("part_type")}
    part_cost = dict(zip(parts["part_id"], parts["unit_cost"]))
    part_lead = dict(zip(parts["part_id"], parts["lead_time_days"]))

    ro_rows = []
    usage_rows = []

    start_dt = datetime(2023, 1, 1)
    end_dt = datetime(2026, 7, 1)

    subsystems = list(FAILURE_MAP.keys())
    subsystem_weights = [0.22, 0.2, 0.18, 0.16, 0.14, 0.10]

    for i in range(1, n_ro + 1):
        ro_id = f"RO{i:06d}"
        subsystem = random.choices(subsystems, weights=subsystem_weights, k=1)[0]
        symptom = random.choice(FAILURE_MAP[subsystem])
        failure_code = FAILURE_CODE[subsystem]

        model = random.choice(AUTOMOTIVE_MODELS)
        age = round(max(0.2, np.random.normal(6.5, 3.5)), 1)
        hours = int(max(100, np.random.normal(15000, 8000)))
        env = random.choice(ENVIRONMENTS)

        created_at = make_timestamp(start_dt, end_dt)
        downtime = round(max(0.5, np.random.normal(5.5, 2.0)), 2)

        # pick associated template
        template = weighted_choice(ASSOCIATIONS[subsystem])
        chosen_types = list(template)

        # optional extra associated part (20%)
        if random.random() < 0.20:
            extra_pool = list(ASSOCIATIONS[subsystem])
            extra = random.choice(random.choice(extra_pool))
            for t in extra if isinstance(extra, list) else [extra]:
                if isinstance(t, str) and t not in chosen_types:
                    chosen_types.append(t)

        # map types -> concrete part_ids
        used_part_ids = []
        for ptype in chosen_types:
            if ptype in part_by_type:
                used_part_ids.append(random.choice(part_by_type[ptype]))

        used_part_ids = list(dict.fromkeys(used_part_ids))  # dedup keep order
        if not used_part_ids:
            continue

        est_cost = 0.0
        for ln, pid in enumerate(used_part_ids, start=1):
            qty = 1 if random.random() < 0.85 else 2
            unit = float(part_cost[pid])
            line_cost = round(qty * unit, 2)
            est_cost += line_cost

            usage_rows.append({
                "ro_id": ro_id,
                "line_num": ln,
                "part_id": pid,
                "part_name": parts.loc[parts["part_id"] == pid, "part_name"].iloc[0],
                "part_type": parts.loc[parts["part_id"] == pid, "part_type"].iloc[0],
                "qty": qty,
                "unit_cost": unit,
                "line_cost": line_cost,
                "lead_time_days": int(part_lead[pid]),
                "is_primary_fix": (ln == 1)
            })

        notes = f"Observed: {symptom}. Inspected {subsystem} subsystem. Replaced primary component and verified normal operation."
        ro_rows.append({
            "ro_id": ro_id,
            "machine_model": model,
            "machine_age_years": age,
            "operating_hours": hours,
            "symptom_text": symptom,
            "failure_code": failure_code,
            "subsystem": subsystem,
            "environment": env,
            "technician_notes": notes,
            "downtime_hours": downtime,
            "estimated_total_parts_cost": round(est_cost, 2),
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S")
        })

    repair_orders = pd.DataFrame(ro_rows).sort_values("ro_id").reset_index(drop=True)
    ro_parts_used = pd.DataFrame(usage_rows).sort_values(["ro_id", "line_num"]).reset_index(drop=True)

    # training_multi_label build
    grouped = ro_parts_used.groupby("ro_id").agg({
        "part_id": lambda s: "|".join(sorted(set(s))),
        "part_type": lambda s: "|".join(sorted(set(s)))
    }).rename(columns={"part_id": "label_all_part_ids", "part_type": "label_all_part_types"}).reset_index()

    train_df = repair_orders.merge(grouped, on="ro_id", how="left")
    train_df["created_at_dt"] = pd.to_datetime(train_df["created_at"])
    train_df = train_df.sort_values("created_at_dt").reset_index(drop=True)

    n = len(train_df)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    split = np.array(["train"] * n)
    split[n_train:n_train+n_val] = "val"
    split[n_train+n_val:] = "test"
    train_df["split"] = split
    train_df = train_df.drop(columns=["created_at_dt"])

    return parts, repair_orders, ro_parts_used, train_df

def main():
    parts, ro, rpu, train = build_datasets(n_ro=1000, seed=42)

    parts.to_csv(os.path.join(paths.RAW_DATA, "parts.csv"), index=False)
    ro.to_csv(os.path.join(paths.RAW_DATA, "repair_orders.csv"), index=False)
    rpu.to_csv(os.path.join(paths.RAW_DATA, "ro_parts_used.csv"), index=False)
    train.to_csv(os.path.join(paths.RAW_DATA, "training_multi_label.csv"), index=False)

    print("=== Automotive synthetic datasets generated ===")
    print(f"parts.csv rows: {len(parts)}")
    print(f"repair_orders.csv rows: {len(ro)}")
    print(f"ro_parts_used.csv rows: {len(rpu)}")
    print(f"training_multi_label.csv rows: {len(train)}")
    print(train['split'].value_counts().to_string())

if __name__ == "__main__":
    main()