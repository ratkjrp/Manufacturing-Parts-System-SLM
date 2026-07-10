import random
import uuid
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from util import paths

# -----------------------------
# Config
# -----------------------------
SEED = 42
N_PARTS = 300
N_REPAIR_ORDERS = 1000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2026, 6, 30)

# Noise controls
TYPO_RATE = 0.08
ABBREV_RATE = 0.12
MISSING_NOTE_RATE = 0.04
MISSING_FAILURE_CODE_RATE = 0.03
SUBSTITUTE_PART_RATE = 0.05

# Class balance controls (head/mid/tail)
HEAD_RATIO = 0.70
MID_RATIO = 0.20
TAIL_RATIO = 0.10

# -----------------------------
# Seed everything
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Domain dictionaries
# -----------------------------
OEMS = ["OmniDrive", "AtlasMotion", "ForgeLine", "HydraCore", "VectorMech"]

MACHINE_MODELS = [
    "CNC-4020", "CNC-5100", "HYD-Press-880", "HYD-Press-940",
    "CONV-X200", "CONV-X350", "PACK-700", "PACK-900",
    "MILL-3AX", "MILL-5AX"
]

ENVIRONMENTS = ["normal", "dusty", "high_humidity", "high_temp", "24x7_duty"]

SUBSYSTEMS = {
    "spindle": ["bearing_set", "spindle_motor", "shaft_seal", "high_temp_grease"],
    "hydraulic": ["seal_kit", "hydraulic_hose", "pump", "fluid_filter", "pressure_valve"],
    "conveyor": ["belt", "idler_roller", "drive_chain", "chain_lube", "tensioner"],
    "electrical": ["contactor", "relay", "fuse", "sensor_prox", "wiring_harness"],
    "cooling": ["fan", "coolant_pump", "radiator_core", "thermostat", "coolant_filter"],
    "pneumatic": ["air_regulator", "solenoid_valve", "air_line", "frl_unit"],
}

FAILURE_PATTERNS = [
    {
        "failure_code": "FC-SPN-17",
        "symptoms": [
            "Spindle overheating at high RPM",
            "Abnormal spindle temperature alarm",
            "Thermal trip during long cycle",
        ],
        "subsystem": "spindle",
        "primary_candidates": ["bearing_set", "spindle_motor", "high_temp_grease"],
        "co_parts": ["shaft_seal", "high_temp_grease"],
        "base_downtime": 5.5,
    },
    {
        "failure_code": "FC-HYD-04",
        "symptoms": [
            "Hydraulic pressure drops after 20 min",
            "Pressure unstable during press cycle",
            "Hydraulic force below setpoint",
        ],
        "subsystem": "hydraulic",
        "primary_candidates": ["seal_kit", "pump", "pressure_valve", "hydraulic_hose"],
        "co_parts": ["fluid_filter", "hydraulic_hose"],
        "base_downtime": 6.0,
    },
    {
        "failure_code": "FC-CON-09",
        "symptoms": [
            "Conveyor slipping under load",
            "Intermittent conveyor jerk",
            "Line speed not maintained",
        ],
        "subsystem": "conveyor",
        "primary_candidates": ["belt", "drive_chain", "tensioner", "idler_roller"],
        "co_parts": ["chain_lube", "idler_roller"],
        "base_downtime": 3.5,
    },
    {
        "failure_code": "FC-ELC-22",
        "symptoms": [
            "Unexpected motor stop fault",
            "Control cabinet overheating trip",
            "Intermittent sensor fault",
        ],
        "subsystem": "electrical",
        "primary_candidates": ["contactor", "relay", "sensor_prox", "wiring_harness", "fuse"],
        "co_parts": ["fuse", "relay"],
        "base_downtime": 2.5,
    },
    {
        "failure_code": "FC-CLG-11",
        "symptoms": [
            "Coolant temperature high",
            "Insufficient coolant flow alarm",
            "Heat exchanger inefficiency warning",
        ],
        "subsystem": "cooling",
        "primary_candidates": ["coolant_pump", "fan", "radiator_core", "thermostat"],
        "co_parts": ["coolant_filter", "fan"],
        "base_downtime": 4.0,
    },
    {
        "failure_code": "FC-PNE-07",
        "symptoms": [
            "Actuator response delayed",
            "Air pressure fluctuation in line",
            "Pneumatic valve not switching",
        ],
        "subsystem": "pneumatic",
        "primary_candidates": ["solenoid_valve", "air_regulator", "frl_unit", "air_line"],
        "co_parts": ["air_line", "frl_unit"],
        "base_downtime": 3.0,
    },
]

PART_CATEGORY_MAP = {
    "bearing_set": ("mechanical", "bearings"),
    "spindle_motor": ("electromechanical", "motors"),
    "shaft_seal": ("mechanical", "seals"),
    "high_temp_grease": ("consumable", "lubricants"),
    "seal_kit": ("mechanical", "seals"),
    "hydraulic_hose": ("hydraulic", "hoses"),
    "pump": ("hydraulic", "pumps"),
    "fluid_filter": ("hydraulic", "filters"),
    "pressure_valve": ("hydraulic", "valves"),
    "belt": ("mechanical", "belts"),
    "idler_roller": ("mechanical", "rollers"),
    "drive_chain": ("mechanical", "chains"),
    "chain_lube": ("consumable", "lubricants"),
    "tensioner": ("mechanical", "tensioners"),
    "contactor": ("electrical", "switchgear"),
    "relay": ("electrical", "switchgear"),
    "fuse": ("electrical", "protection"),
    "sensor_prox": ("electrical", "sensors"),
    "wiring_harness": ("electrical", "cabling"),
    "fan": ("cooling", "air_movement"),
    "coolant_pump": ("cooling", "pumps"),
    "radiator_core": ("cooling", "heat_exchange"),
    "thermostat": ("cooling", "controls"),
    "coolant_filter": ("cooling", "filters"),
    "air_regulator": ("pneumatic", "regulators"),
    "solenoid_valve": ("pneumatic", "valves"),
    "air_line": ("pneumatic", "hoses"),
    "frl_unit": ("pneumatic", "conditioning"),
}

# Optional substitutes/supersessions
SUBSTITUTE_MAP = {
    "bearing_set": ["bearing_set_v2"],
    "seal_kit": ["seal_kit_plus"],
    "hydraulic_hose": ["hydraulic_hose_revB"],
    "contactor": ["contactor_gen2"],
    "coolant_pump": ["coolant_pump_x"],
}

ABBREV_REPLACEMENTS = {
    "replaced": "rplcd",
    "pressure": "press",
    "temperature": "temp",
    "hydraulic": "hyd",
    "conveyor": "conv",
    "spindle": "spndl",
    "bearing": "brg",
    "sensor": "snsr",
    "filter": "fltr",
    "verified": "vrfd",
}

TYPO_REPLACEMENTS = {
    "bearing": "bering",
    "hydraulic": "hydralic",
    "temperature": "temprature",
    "pressure": "presure",
    "conveyor": "convyor",
    "coolant": "colant",
    "sensor": "senser",
}


# -----------------------------
# Helper functions
# -----------------------------
def random_date(start_dt, end_dt):
    delta = end_dt - start_dt
    return start_dt + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def choose_machine_for_subsystem(subsystem):
    if subsystem in ["spindle", "cooling"]:
        candidates = ["CNC-4020", "CNC-5100", "MILL-3AX", "MILL-5AX"]
    elif subsystem == "hydraulic":
        candidates = ["HYD-Press-880", "HYD-Press-940"]
    elif subsystem == "conveyor":
        candidates = ["CONV-X200", "CONV-X350", "PACK-700", "PACK-900"]
    elif subsystem == "electrical":
        candidates = MACHINE_MODELS
    elif subsystem == "pneumatic":
        candidates = ["PACK-700", "PACK-900", "CONV-X200", "CONV-X350"]
    else:
        candidates = MACHINE_MODELS
    return random.choice(candidates)


def inject_abbrev(text):
    words = text.split()
    out = []
    for w in words:
        key = w.lower().strip(".,")
        if key in ABBREV_REPLACEMENTS and random.random() < 0.3:
            out.append(ABBREV_REPLACEMENTS[key])
        else:
            out.append(w)
    return " ".join(out)


def inject_typo(text):
    out = text
    for k, v in TYPO_REPLACEMENTS.items():
        if k in out.lower() and random.random() < 0.35:
            # naive case-insensitive replace
            idx = out.lower().find(k)
            if idx >= 0:
                out = out[:idx] + v + out[idx+len(k):]
    return out


def noisy_note(base_note):
    txt = base_note
    if random.random() < ABBREV_RATE:
        txt = inject_abbrev(txt)
    if random.random() < TYPO_RATE:
        txt = inject_typo(txt)
    return txt


def make_part_id(i):
    return f"P-{1000 + i}"


def weighted_part_frequency(parts_df):
    """
    Assign each part to head/mid/tail buckets and create a sampling weight.
    """
    n = len(parts_df)
    idx = np.arange(n)
    np.random.shuffle(idx)

    head_n = int(n * HEAD_RATIO)
    mid_n = int(n * MID_RATIO)
    tail_n = n - head_n - mid_n

    bucket = np.array(["tail"] * n, dtype=object)
    bucket[idx[:head_n]] = "head"
    bucket[idx[head_n:head_n + mid_n]] = "mid"

    # Higher weight = appears more often
    weights = np.where(bucket == "head", 1.0, np.where(bucket == "mid", 0.35, 0.10))
    weights = weights / weights.sum()
    return bucket, weights


# -----------------------------
# Build parts catalog
# -----------------------------
def generate_parts_catalog(n_parts=N_PARTS):
    base_types = list(PART_CATEGORY_MAP.keys())
    rows = []
    for i in range(n_parts):
        ptype = random.choice(base_types)
        cat, subcat = PART_CATEGORY_MAP[ptype]
        oem = random.choice(OEMS)

        part_id = make_part_id(i + 1)
        part_name = f"{ptype.upper()}-{random.randint(10,99)}"
        unit_cost = round(np.clip(np.random.normal(loc=120, scale=80), 8, 1200), 2)
        lead_time_days = int(np.clip(np.random.normal(loc=7, scale=4), 1, 45))

        # compatibility
        if ptype in ["seal_kit", "pump", "pressure_valve", "hydraulic_hose", "fluid_filter"]:
            compat = ["HYD-Press-880", "HYD-Press-940"]
        elif ptype in ["belt", "idler_roller", "drive_chain", "tensioner", "chain_lube"]:
            compat = ["CONV-X200", "CONV-X350", "PACK-700", "PACK-900"]
        elif ptype in ["bearing_set", "spindle_motor", "shaft_seal", "high_temp_grease"]:
            compat = ["CNC-4020", "CNC-5100", "MILL-3AX", "MILL-5AX"]
        elif ptype in ["fan", "coolant_pump", "radiator_core", "thermostat", "coolant_filter"]:
            compat = ["CNC-4020", "CNC-5100", "MILL-3AX", "MILL-5AX", "HYD-Press-940"]
        else:
            compat = random.sample(MACHINE_MODELS, k=random.randint(3, len(MACHINE_MODELS)))

        rows.append({
            "part_id": part_id,
            "part_type": ptype,
            "part_name": part_name,
            "category": cat,
            "subcategory": subcat,
            "oem": oem,
            "unit_cost": unit_cost,
            "lead_time_days": lead_time_days,
            "compatible_models": "|".join(sorted(set(compat))),
            "is_active": random.random() > 0.02,
        })

    parts = pd.DataFrame(rows)

    # Add substitute/superseded pseudo-parts for realism
    add_rows = []
    for src_type, subs in SUBSTITUTE_MAP.items():
        if random.random() < 0.9:
            sub_type = random.choice(subs)
            cat, subcat = PART_CATEGORY_MAP.get(src_type, ("misc", "misc"))
            add_rows.append({
                "part_id": make_part_id(len(parts) + len(add_rows) + 1),
                "part_type": sub_type,
                "part_name": f"{sub_type.upper()}-{random.randint(10,99)}",
                "category": cat,
                "subcategory": subcat,
                "oem": random.choice(OEMS),
                "unit_cost": round(np.clip(np.random.normal(loc=140, scale=70), 10, 1500), 2),
                "lead_time_days": int(np.clip(np.random.normal(loc=9, scale=5), 1, 60)),
                "compatible_models": "|".join(random.sample(MACHINE_MODELS, k=random.randint(2, 5))),
                "is_active": True,
            })

    if add_rows:
        parts = pd.concat([parts, pd.DataFrame(add_rows)], ignore_index=True)

    # assign usage bucket + sampling weights
    bucket, weights = weighted_part_frequency(parts)
    parts["usage_bucket"] = bucket
    parts["sampling_weight"] = weights
    return parts


# -----------------------------
# Build helper indexes
# -----------------------------
def build_part_index(parts_df):
    by_type = {}
    for ptype, grp in parts_df.groupby("part_type"):
        by_type[ptype] = grp.to_dict("records")
    return by_type


def pick_part_by_type(part_index, ptype, machine_model=None):
    if ptype not in part_index:
        return None
    candidates = part_index[ptype]
    if machine_model:
        compat = [p for p in candidates if machine_model in p["compatible_models"].split("|")]
        if compat:
            candidates = compat
    return random.choice(candidates) if candidates else None


def maybe_substitute_type(ptype):
    if ptype in SUBSTITUTE_MAP and random.random() < SUBSTITUTE_PART_RATE:
        return random.choice(SUBSTITUTE_MAP[ptype])
    return ptype


# -----------------------------
# Generate repair orders + line items
# -----------------------------
def generate_repair_orders(parts_df, n_orders=N_REPAIR_ORDERS):
    part_index = build_part_index(parts_df)

    ro_rows = []
    ro_parts_rows = []

    for i in range(1, n_orders + 1):
        ro_id = f"RO{i:06d}"

        pattern = random.choice(FAILURE_PATTERNS)
        subsystem = pattern["subsystem"]
        machine_model = choose_machine_for_subsystem(subsystem)
        machine_age = round(np.clip(np.random.normal(7, 3), 0.2, 20), 1)

        # operating hours correlated with age
        operating_hours = int(np.clip(np.random.normal(machine_age * 2200, 1800), 100, 50000))

        symptom_text = random.choice(pattern["symptoms"])
        failure_code = pattern["failure_code"]

        # Introduce occasional missing failure code
        if random.random() < MISSING_FAILURE_CODE_RATE:
            failure_code = None

        env = random.choice(ENVIRONMENTS)

        # Choose primary failed part
        primary_type = random.choice(pattern["primary_candidates"])

        # Age/hours effects
        if operating_hours > 18000 and "bearing_set" in pattern["primary_candidates"] and random.random() < 0.2:
            primary_type = "bearing_set"
        if env in ["high_humidity", "dusty"] and "seal_kit" in pattern["primary_candidates"] and random.random() < 0.25:
            primary_type = "seal_kit"

        primary_type = maybe_substitute_type(primary_type)
        primary_part = pick_part_by_type(part_index, primary_type, machine_model)

        # Fallback if substitute type doesn't exist
        if primary_part is None:
            # try original
            base_primary = random.choice(pattern["primary_candidates"])
            primary_part = pick_part_by_type(part_index, base_primary, machine_model)

        used_parts = []

        if primary_part:
            qty = 1 if random.random() < 0.85 else random.randint(2, 3)
            used_parts.append((primary_part, qty, 1))

        # Co-parts
        for cp in pattern["co_parts"]:
            if random.random() < 0.55:
                cp_eff = maybe_substitute_type(cp)
                p = pick_part_by_type(part_index, cp_eff, machine_model)
                if p is not None:
                    qty = 1 if random.random() < 0.9 else random.randint(2, 4)
                    used_parts.append((p, qty, 0))

        # Generic consumables/fasteners with low probability
        if random.random() < 0.15:
            extra_type = random.choice(["high_temp_grease", "chain_lube", "fluid_filter", "fuse"])
            p = pick_part_by_type(part_index, extra_type, machine_model)
            if p is not None:
                used_parts.append((p, 1, 0))

        # Remove duplicates by part_id (sum qty, keep primary if any)
        merged = {}
        for p, qty, is_primary in used_parts:
            pid = p["part_id"]
            if pid not in merged:
                merged[pid] = {"part": p, "qty": qty, "is_primary_fix": is_primary}
            else:
                merged[pid]["qty"] += qty
                merged[pid]["is_primary_fix"] = max(merged[pid]["is_primary_fix"], is_primary)

        used_parts = [(v["part"], v["qty"], v["is_primary_fix"]) for v in merged.values()]

        # Compute downtime: base + part lead + severity proxy
        part_cost_sum = sum(p["unit_cost"] * qty for p, qty, _ in used_parts) if used_parts else 0
        avg_lead = np.mean([p["lead_time_days"] for p, _, _ in used_parts]) if used_parts else 2
        severity = 1.0 + (0.2 if operating_hours > 20000 else 0) + (0.15 if env in ["high_temp", "24x7_duty"] else 0)
        downtime = round(max(0.5, np.random.normal(pattern["base_downtime"] * severity + avg_lead * 0.08, 1.2)), 2)

        base_note = (
            f"Observed: {symptom_text}. "
            f"Inspected {subsystem} subsystem. "
            f"Replaced primary component and verified normal operation."
        )
        tech_note = noisy_note(base_note)
        if random.random() < MISSING_NOTE_RATE:
            tech_note = None

        created_at = random_date(START_DATE, END_DATE)

        ro_rows.append({
            "ro_id": ro_id,
            "machine_model": machine_model,
            "machine_age_years": machine_age,
            "operating_hours": operating_hours,
            "symptom_text": symptom_text,
            "failure_code": failure_code,
            "subsystem": subsystem,
            "environment": env,
            "technician_notes": tech_note,
            "downtime_hours": downtime,
            "estimated_total_parts_cost": round(part_cost_sum, 2),
            "created_at": created_at.isoformat(),
        })

        for line_num, (p, qty, is_primary) in enumerate(used_parts, start=1):
            line_cost = round(p["unit_cost"] * qty, 2)
            ro_parts_rows.append({
                "ro_id": ro_id,
                "line_num": line_num,
                "part_id": p["part_id"],
                "part_name": p["part_name"],
                "part_type": p["part_type"],
                "qty": qty,
                "is_primary_fix": bool(is_primary),
                "unit_cost": p["unit_cost"],
                "line_cost": line_cost,
                "lead_time_days": p["lead_time_days"],
            })

    return pd.DataFrame(ro_rows), pd.DataFrame(ro_parts_rows)


# -----------------------------
# Optional: training datasets
# -----------------------------
def build_training_views(repair_orders_df, ro_parts_df):
    """
    Creates:
    1) single-label dataset (primary part)
    2) multi-label dataset (all parts as pipe-separated list)
    """
    # Primary label
    primary = ro_parts_df[ro_parts_df["is_primary_fix"] == True].copy()
    primary = primary.sort_values(["ro_id", "line_num"]).drop_duplicates("ro_id")

    single = repair_orders_df.merge(
        primary[["ro_id", "part_id", "part_type"]],
        on="ro_id",
        how="left"
    ).rename(columns={"part_id": "label_primary_part_id", "part_type": "label_primary_part_type"})

    # Multi-label
    all_parts = ro_parts_df.groupby("ro_id")["part_id"].apply(lambda s: "|".join(sorted(set(s)))).reset_index()
    all_types = ro_parts_df.groupby("ro_id")["part_type"].apply(lambda s: "|".join(sorted(set(s)))).reset_index()
    multi = repair_orders_df.merge(all_parts, on="ro_id", how="left").merge(all_types, on="ro_id", how="left")
    multi = multi.rename(columns={"part_id": "label_all_part_ids", "part_type": "label_all_part_types"})

    return single, multi


# -----------------------------
# Main
# -----------------------------
def main():
    parts_df = generate_parts_catalog(N_PARTS)
    repair_orders_df, ro_parts_df = generate_repair_orders(parts_df, N_REPAIR_ORDERS)

    single_label_df, multi_label_df = build_training_views(repair_orders_df, ro_parts_df)

    # Split by time (better for leakage prevention)
    repair_orders_df["created_at"] = pd.to_datetime(repair_orders_df["created_at"])
    cutoff_train = repair_orders_df["created_at"].quantile(0.8)
    cutoff_val = repair_orders_df["created_at"].quantile(0.9)

    def split_tag(ts):
        if ts <= cutoff_train:
            return "train"
        elif ts <= cutoff_val:
            return "val"
        return "test"

    split_map = repair_orders_df[["ro_id", "created_at"]].copy()
    split_map["split"] = split_map["created_at"].apply(split_tag)

    # write outputs
    parts_df.to_csv(os.path.join(paths.RAW_DATA, "parts.csv"), index=False)
    repair_orders_df.to_csv(os.path.join(paths.RAW_DATA, "repair_orders.csv"), index=False)
    ro_parts_df.to_csv(os.path.join(paths.RAW_DATA, "ro_parts_used.csv"), index=False)

    single_out = single_label_df.merge(split_map[["ro_id", "split"]], on="ro_id", how="left")
    multi_out = multi_label_df.merge(split_map[["ro_id", "split"]], on="ro_id", how="left")

    single_out.to_csv(os.path.join(paths.RAW_DATA, "training_single_label.csv"), index=False)
    multi_out.to_csv(os.path.join(paths.RAW_DATA, "training_multi_label.csv"), index=False)

    # basic stats
    print("Generated files:")
    print(" - parts.csv")
    print(" - repair_orders.csv")
    print(" - ro_parts_used.csv")
    print(" - training_single_label.csv")
    print(" - training_multi_label.csv")
    print("\nCounts:")
    print(f"parts: {len(parts_df):,}")
    print(f"repair_orders: {len(repair_orders_df):,}")
    print(f"ro_parts_used rows: {len(ro_parts_df):,}")
    print("\nSplit distribution:")
    print(single_out['split'].value_counts(dropna=False))


if __name__ == "__main__":
    main()