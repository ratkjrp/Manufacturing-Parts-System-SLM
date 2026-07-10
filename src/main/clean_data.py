#!/usr/bin/env python3
"""
Clean manufacturing maintenance datasets for robust train/val/test workflows.

Default behavior (no CLI args):
- Reads from: data/raw
- Writes to: data/processed

So you can run:
    python src/main/clean_data.py

Optional overrides:
    python src/main/clean_data.py --input_dir ./some/input --output_dir ./some/output
"""

import argparse
import os
import re
import unicodedata
from typing import List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Utility helpers
# -----------------------------
def normalize_text(s: object) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_token(s: object) -> str:
    s = normalize_text(s).lower()
    # keep alnum, _, -, |
    s = re.sub(r"[^a-z0-9_\-\| ]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s.strip("_")


def parse_bool(v: object, default=True) -> bool:
    if pd.isna(v):
        return default
    x = str(v).strip().lower()
    if x in {"true", "1", "yes", "y", "t"}:
        return True
    if x in {"false", "0", "no", "n", "f"}:
        return False
    return default


def safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def split_pipe_unique(s: object) -> List[str]:
    if pd.isna(s):
        return []
    vals = [normalize_text(x) for x in str(s).split("|")]
    vals = [v for v in vals if v]
    # preserve order unique
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def join_pipe(vals: List[str]) -> str:
    return "|".join(vals)


def resolve_default_dirs():
    """
    Resolve project-root-relative default folders regardless of current working directory.
    clean_data.py is expected at: src/main/clean_data.py
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))       # .../src/main
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    default_input = os.path.join(project_root, "data", "raw")
    default_output = os.path.join(project_root, "data", "processed")
    return default_input, default_output


# -----------------------------
# Cleaning functions
# -----------------------------
def clean_parts(parts: pd.DataFrame) -> pd.DataFrame:
    df = parts.copy()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Standardize key fields
    df["part_id"] = df["part_id"].astype(str).str.strip().str.upper()
    df["part_type"] = df["part_type"].map(normalize_token)
    df["part_name"] = df["part_name"].map(normalize_text).str.upper()
    df["category"] = df["category"].map(normalize_token)
    df["subcategory"] = df["subcategory"].map(normalize_token)
    df["oem"] = df["oem"].map(normalize_text)
    df["compatible_models"] = df["compatible_models"].map(lambda x: join_pipe(split_pipe_unique(x)))
    df["usage_bucket"] = df["usage_bucket"].map(normalize_token)

    # Numerics
    for c in ["unit_cost", "lead_time_days", "sampling_weight"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Booleans
    if "is_active" in df.columns:
        df["is_active"] = df["is_active"].map(parse_bool)

    # Basic clamps
    if "unit_cost" in df.columns:
        df["unit_cost"] = df["unit_cost"].clip(lower=0)
    if "lead_time_days" in df.columns:
        df["lead_time_days"] = df["lead_time_days"].clip(lower=0)
    if "sampling_weight" in df.columns:
        df["sampling_weight"] = df["sampling_weight"].clip(lower=0)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Keep latest occurrence per part_id (if duplicate IDs exist)
    df = df.drop_duplicates(subset=["part_id"], keep="last")

    # Deterministic ordering
    df = df.sort_values("part_id").reset_index(drop=True)
    return df


def clean_repair_orders(ro: pd.DataFrame) -> pd.DataFrame:
    df = ro.copy()
    df.columns = [c.strip() for c in df.columns]

    # IDs / enums
    df["ro_id"] = df["ro_id"].astype(str).str.strip().str.upper()
    df["machine_model"] = df["machine_model"].map(normalize_text)
    df["symptom_text"] = df["symptom_text"].map(normalize_text)
    df["failure_code"] = df["failure_code"].map(lambda x: normalize_text(x).upper())
    df["subsystem"] = df["subsystem"].map(normalize_token)
    df["environment"] = df["environment"].map(normalize_token)
    df["technician_notes"] = df["technician_notes"].map(normalize_text)

    # Fill missing failure_code by subsystem (simple heuristic)
    subsystem_to_code = {
        "spindle": "FC-SPN-17",
        "hydraulic": "FC-HYD-04",
        "pneumatic": "FC-PNE-07",
        "cooling": "FC-CLG-11",
        "electrical": "FC-ELC-22",
        "conveyor": "FC-CON-09",
    }
    missing_fc = df["failure_code"].isna() | (df["failure_code"] == "")
    df.loc[missing_fc, "failure_code"] = df.loc[missing_fc, "subsystem"].map(subsystem_to_code).fillna("UNKNOWN")

    # Numerics
    num_cols = ["machine_age_years", "operating_hours", "downtime_hours", "estimated_total_parts_cost"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clamp invalids
    df["machine_age_years"] = df["machine_age_years"].clip(lower=0)
    df["operating_hours"] = df["operating_hours"].clip(lower=0)
    df["downtime_hours"] = df["downtime_hours"].clip(lower=0)
    df["estimated_total_parts_cost"] = df["estimated_total_parts_cost"].clip(lower=0)

    # Datetime
    df["created_at"] = safe_datetime(df["created_at"])

    # Remove rows with no ro_id
    df = df[df["ro_id"].notna() & (df["ro_id"] != "")]

    # Keep latest by ro_id
    df = df.sort_values("created_at").drop_duplicates(subset=["ro_id"], keep="last")

    df = df.sort_values("ro_id").reset_index(drop=True)
    return df


def clean_ro_parts_used(rpu: pd.DataFrame, parts_clean: pd.DataFrame) -> pd.DataFrame:
    df = rpu.copy()
    df.columns = [c.strip() for c in df.columns]

    df["ro_id"] = df["ro_id"].astype(str).str.strip().str.upper()
    df["part_id"] = df["part_id"].astype(str).str.strip().str.upper()
    df["part_name"] = df["part_name"].map(normalize_text).str.upper()
    df["part_type"] = df["part_type"].map(normalize_token)
    df["is_primary_fix"] = df["is_primary_fix"].map(parse_bool)

    for c in ["line_num", "qty", "unit_cost", "line_cost", "lead_time_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill sensible defaults
    df["line_num"] = df["line_num"].fillna(1).astype(int)
    df["qty"] = df["qty"].fillna(1).clip(lower=1).astype(int)
    df["unit_cost"] = df["unit_cost"].fillna(0).clip(lower=0)
    df["lead_time_days"] = df["lead_time_days"].fillna(0).clip(lower=0)

    # Recompute line_cost to enforce consistency
    df["line_cost"] = (df["qty"] * df["unit_cost"]).round(2)

    # Fix part_name/type from authoritative parts table when available
    parts_lookup = parts_clean.set_index("part_id")[["part_name", "part_type", "unit_cost", "lead_time_days"]]
    matched = df["part_id"].isin(parts_lookup.index)

    df.loc[matched, "part_name"] = df.loc[matched, "part_id"].map(parts_lookup["part_name"])
    df.loc[matched, "part_type"] = df.loc[matched, "part_id"].map(parts_lookup["part_type"])

    # If unit_cost missing/zero and part is known, use parts master
    needs_cost = matched & ((df["unit_cost"].isna()) | (df["unit_cost"] <= 0))
    df.loc[needs_cost, "unit_cost"] = df.loc[needs_cost, "part_id"].map(parts_lookup["unit_cost"])
    df.loc[needs_cost, "line_cost"] = (df.loc[needs_cost, "qty"] * df.loc[needs_cost, "unit_cost"]).round(2)

    # If lead time missing/zero and part is known, use parts master
    needs_lt = matched & ((df["lead_time_days"].isna()) | (df["lead_time_days"] <= 0))
    df.loc[needs_lt, "lead_time_days"] = df.loc[needs_lt, "part_id"].map(parts_lookup["lead_time_days"])

    # Drop exact duplicates
    df = df.drop_duplicates()

    # De-dup within (ro_id, part_id, line_num): keep last
    df = df.sort_values(["ro_id", "line_num"]).drop_duplicates(subset=["ro_id", "part_id", "line_num"], keep="last")

    # Re-sequence line_num within each RO
    df = df.sort_values(["ro_id", "line_num", "part_id"]).reset_index(drop=True)
    df["line_num"] = df.groupby("ro_id").cumcount() + 1

    return df


def rebuild_training_from_ro(
    ro_clean: pd.DataFrame, rpu_clean: pd.DataFrame, existing_train: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    # Aggregate labels from parts used
    agg = (
        rpu_clean.groupby("ro_id")
        .agg(
            label_all_part_ids=("part_id", lambda s: join_pipe(sorted(set(map(str, s))))),
            label_all_part_types=("part_type", lambda s: join_pipe(sorted(set(map(str, s))))),
            observed_parts_cost=("line_cost", "sum"),
        )
        .reset_index()
    )

    out = ro_clean.merge(agg, on="ro_id", how="left")

    # Keep existing split if provided and valid, else assign time-based split
    if existing_train is not None and "split" in existing_train.columns:
        split_map = existing_train[["ro_id", "split"]].copy()
        split_map["ro_id"] = split_map["ro_id"].astype(str).str.strip().str.upper()
        split_map["split"] = split_map["split"].map(normalize_token)
        out = out.merge(split_map, on="ro_id", how="left", suffixes=("", "_old"))
        out["split"] = out["split"].where(out["split"].isin(["train", "val", "test"]), np.nan)

    if "split" not in out.columns:
        out["split"] = np.nan

    # Time-based fallback split
    missing = out["split"].isna()
    if missing.any():
        tmp = out.loc[missing].sort_values("created_at")
        n = len(tmp)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        idx = tmp.index.to_list()

        out.loc[idx[:n_train], "split"] = "train"
        out.loc[idx[n_train:n_train + n_val], "split"] = "val"
        out.loc[idx[n_train + n_val:], "split"] = "test"

    # Final columns in ML-friendly order
    cols = [
        "ro_id", "machine_model", "machine_age_years", "operating_hours",
        "symptom_text", "failure_code", "subsystem", "environment",
        "technician_notes", "downtime_hours", "estimated_total_parts_cost",
        "created_at", "label_all_part_ids", "label_all_part_types", "split"
    ]
    out = out[cols].sort_values("ro_id").reset_index(drop=True)
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    default_input_dir, default_output_dir = resolve_default_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default=".",
        help="Directory with raw CSV files (default: current directory)"
    )
    parser.add_argument(
        "--output_dir",
        default="./cleaned",
        help="Directory to write cleaned CSVs (default: ./cleaned)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parts_path = os.path.join(args.input_dir, "parts.csv")
    ro_path = os.path.join(args.input_dir, "repair_orders.csv")
    rpu_path = os.path.join(args.input_dir, "ro_parts_used.csv")
    train_path = os.path.join(args.input_dir, "training_multi_label.csv")

    required = [parts_path, ro_path, rpu_path]
    missing_required = [p for p in required if not os.path.exists(p)]
    if missing_required:
        raise FileNotFoundError(
            "Missing required input files:\n- " + "\n- ".join(missing_required) +
            f"\n\nExpected input_dir: {args.input_dir}"
        )

    # Load
    parts = pd.read_csv(parts_path)
    ro = pd.read_csv(ro_path)
    rpu = pd.read_csv(rpu_path)
    train_existing = pd.read_csv(train_path) if os.path.exists(train_path) else None

    # Clean
    parts_clean = clean_parts(parts)
    ro_clean = clean_repair_orders(ro)
    rpu_clean = clean_ro_parts_used(rpu, parts_clean)

    # Keep only ROs that exist in cleaned repair_orders
    rpu_clean = rpu_clean[rpu_clean["ro_id"].isin(set(ro_clean["ro_id"]))].copy()

    # Rebuild training table from cleaned sources
    train_rebuilt = rebuild_training_from_ro(ro_clean, rpu_clean, train_existing)

    # Optional: clean existing training file minimally if present
    if train_existing is not None:
        te = train_existing.copy()
        te.columns = [c.strip() for c in te.columns]
        te["ro_id"] = te["ro_id"].astype(str).str.strip().str.upper()
        if "split" in te.columns:
            te["split"] = te["split"].map(normalize_token)
            te["split"] = te["split"].where(te["split"].isin(["train", "val", "test"]), "train")
        te["created_at"] = safe_datetime(te["created_at"])
        cleaned_training_existing = te
    else:
        cleaned_training_existing = train_rebuilt.copy()

    # Write cleaned base files
    parts_clean.to_csv(os.path.join(args.output_dir, "cleaned_parts.csv"), index=False)
    ro_clean.to_csv(os.path.join(args.output_dir, "cleaned_repair_orders.csv"), index=False)
    rpu_clean.to_csv(os.path.join(args.output_dir, "cleaned_ro_parts_used.csv"), index=False)
    cleaned_training_existing.to_csv(os.path.join(args.output_dir, "cleaned_training_multi_label.csv"), index=False)
    train_rebuilt.to_csv(os.path.join(args.output_dir, "cleaned_training_multi_label_rebuilt.csv"), index=False)

    # Export train/val/test from rebuilt
    train_rebuilt[train_rebuilt["split"] == "train"].to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    train_rebuilt[train_rebuilt["split"] == "val"].to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    train_rebuilt[train_rebuilt["split"] == "test"].to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    # Quick QA summary
    print("=== Cleaning complete ===")
    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"parts: {len(parts)} -> {len(parts_clean)}")
    print(f"repair_orders: {len(ro)} -> {len(ro_clean)}")
    print(f"ro_parts_used: {len(rpu)} -> {len(rpu_clean)}")
    print(f"training_rebuilt: {len(train_rebuilt)} rows")
    print("Split counts:")
    print(train_rebuilt["split"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()