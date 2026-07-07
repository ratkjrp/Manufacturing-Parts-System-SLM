"""
Builds single_var_train.jsonl and single_var_val.jsonl for fine-tuning
SmolLM3 on single-part recommendations.

Reads:
    parts.csv                  - part_id -> part_name/type lookup
    training_single_label.csv  - repair orders + primary part label + split

Writes:
    single_var_train.jsonl
    single_var_val.jsonl

INPUT FIELDS USED (available at intake, before repair):
    machine_model, machine_age_years, operating_hours, symptom_text,
    subsystem, environment, failure_code

FIELDS EXCLUDED FROM INPUT (only known after the repair — including them
would leak the answer / train the model on info it won't have at inference):
    technician_notes, downtime_hours, estimated_total_parts_cost

ASSUMPTION: failure_code is treated as an intake-time diagnostic code.
If your system only assigns it after a technician diagnoses the issue,
remove the failure_code line from build_user_message() below.

Usage:
    python build_chat_single.py --data_dir /mnt/user-data/uploads --output_dir .
"""

import argparse
import csv
import json
import os

SYSTEM_PROMPT = (
    "You are a parts recommendation service for a manufacturing dealer " 
    "parts management system. Based on the item requested to be ordered " 
    "in addition to their model, type, and reported symptom, recommend other "
    "parts that may be in association with that requested item."
)


def load_parts_lookup(parts_csv_path):
    lookup = {}
    with open(parts_csv_path) as f:
        for row in csv.DictReader(f):
            lookup[row["part_id"]] = row
    return lookup


def build_user_message(row):
    return (
        f"Machine: {row['machine_model']} (age: {row['machine_age_years']} years, "
        f"{row['operating_hours']} operating hours)\n"
        f"Subsystem: {row['subsystem']}\n"
        f"Environment: {row['environment']}\n"
        f"Failure code: {row['failure_code'] or 'not assigned'}\n"
        f"Reported symptom: {row['symptom_text']}\n\n"
        f"What part(s) are likely needed for this repair?"
    )


def build_assistant_message(row, parts_lookup):
    part = parts_lookup.get(row["label_primary_part_id"])
    part_name = part["part_name"] if part else row["label_primary_part_id"]
    return (
        f"Recommended part: {part_name} "
        f"({row['label_primary_part_id']}, type: {row['label_primary_part_type']})."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    parts_lookup = load_parts_lookup(os.path.join(args.data_dir, "parts.csv"))
    input_csv = os.path.join(args.data_dir, "training_single_label.csv")

    out_paths = {
        "train": os.path.join(args.output_dir, "single_var_train.jsonl"),
        "val": os.path.join(args.output_dir, "single_var_val.jsonl"),
    }
    # test split is skipped here — keep it held out for evaluation later
    out_files = {split: open(path, "w") for split, path in out_paths.items()}
    counts = {split: 0 for split in out_paths}

    with open(input_csv) as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in out_files:
                continue  # skips "test" rows

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(row)},
                    {"role": "assistant", "content": build_assistant_message(row, parts_lookup)},
                ]
            }
            out_files[split].write(json.dumps(example) + "\n")
            counts[split] += 1

    for f in out_files.values():
        f.close()

    for split, path in out_paths.items():
        print(f"Wrote {counts[split]} examples to {path}")


if __name__ == "__main__":
    main()