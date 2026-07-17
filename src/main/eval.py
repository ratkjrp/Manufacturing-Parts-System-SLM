"""
evaluate.py — Evaluate the fine-tuned parts-recommendation LoRA adapter.
NO extra dependencies (no matplotlib). Uses only pandas/torch/transformers/peft,
which you already have installed.

Outputs (in src/output/eval/):
  - Prints precision / recall / F1 / exact-match to the console
  - metrics_summary.csv     -> the 4 headline numbers (open in Excel, make a bar chart)
  - per_example_results.csv -> row-by-row pred vs gold, so you can inspect quality
  - loss_history.csv        -> train/eval loss over steps (if trainer_state.json exists)
"""

import os
import csv
import json
import re

os.environ["HF_ENDPOINT"] = "https://infyartifactory.jfrog.io/artifactory/api/huggingfaceml/huggingface-remote"
hf_token = os.environ.get("HF_TOKEN")

from util import paths
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -- Config ------------------------------------------------------------------
BASE     = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER  = os.path.join(paths.SRC, "output", "smollm3-finetuned")
VAL_CSV  = os.path.join(paths.PROC_DATA, "val.csv")
OUT_DIR  = os.path.join(paths.SRC, "output", "eval")
MAX_NEW_TOKENS = 120
LIMIT    = None    # set to e.g. 20 for a quick test; None = all rows

SYSTEM_PROMPT = (
    "You are a parts recommendation service for a manufacturing dealer "
    "parts management system. Based on the item requested to be ordered "
    "in addition to their model, type, and reported symptom, recommend other "
    "parts that may be in association with that requested item."
)

os.makedirs(OUT_DIR, exist_ok=True)

# -- Same user-message builder as training (keep in sync!) -------------------
def build_user_msg(row):
    return (
        f"Machine: {row['machine_model']}, "
        f"Age: {row['machine_age_years']} years, "
        f"Operating hours: {row['operating_hours']}, "
        f"Subsystem: {row['subsystem']}, "
        f"Environment: {row['environment']}. "
        f"Symptom: {row['symptom_text']}"
    )

def normalize_parts(text):
    """Comma-separated parts string -> clean set of lowercase names."""
    text = re.sub(r"(?i)recommended parts\s*:", "", text)
    text = text.strip().splitlines()[0] if text.strip() else ""
    items = [p.strip().lower() for p in text.split(",")]
    return {p for p in items if p}

def true_parts_set(row):
    return normalize_parts(str(row["label_all_part_types"]).replace("|", ","))

def prf1(pred, gold):
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    inter = len(pred & gold)
    p = inter / len(pred)
    r = inter / len(gold)
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f

# -- Load model --------------------------------------------------------------
print("Loading base model + adapter...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, token=hf_token)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("Loaded.\n")

# -- Run over validation set -------------------------------------------------
df = pd.read_csv(VAL_CSV)
if LIMIT:
    df = df.head(LIMIT)

precisions, recalls, f1s, exact = [], [], [], []
rows_out = []

for i, (_, row) in enumerate(df.iterrows()):
    user_msg = build_user_msg(row)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    pred = normalize_parts(generated)
    gold = true_parts_set(row)
    p, r, f = prf1(pred, gold)
    precisions.append(p); recalls.append(r); f1s.append(f)
    exact.append(1.0 if pred == gold else 0.0)
    rows_out.append({
        "pred": ", ".join(sorted(pred)),
        "gold": ", ".join(sorted(gold)),
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
    })
    print(f"[{i+1}/{len(df)}]  P={p:.2f} R={r:.2f} F1={f:.2f}")

def avg(xs): return sum(xs) / len(xs) if xs else 0.0
macro_p, macro_r, macro_f, acc = avg(precisions), avg(recalls), avg(f1s), avg(exact)

print("\n==================  RESULTS  ==================")
print(f"Examples evaluated : {len(df)}")
print(f"Precision (avg)    : {macro_p:.3f}")
print(f"Recall    (avg)    : {macro_r:.3f}")
print(f"F1        (avg)    : {macro_f:.3f}")
print(f"Exact-match acc    : {acc:.3f}")
print("===============================================\n")

# -- Write CSVs (no matplotlib needed) ---------------------------------------
with open(os.path.join(OUT_DIR, "metrics_summary.csv"), "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["metric", "score"])
    w.writerow(["Precision", round(macro_p, 3)])
    w.writerow(["Recall", round(macro_r, 3)])
    w.writerow(["F1", round(macro_f, 3)])
    w.writerow(["Exact match", round(acc, 3)])
print("Saved metrics_summary.csv")

pd.DataFrame(rows_out).to_csv(os.path.join(OUT_DIR, "per_example_results.csv"), index=False)
print("Saved per_example_results.csv")

def find_trainer_state(adapter_dir):
    direct = os.path.join(adapter_dir, "trainer_state.json")
    if os.path.exists(direct):
        return direct
    if os.path.isdir(adapter_dir):
        for name in sorted(os.listdir(adapter_dir)):
            cand = os.path.join(adapter_dir, name, "trainer_state.json")
            if os.path.exists(cand):
                return cand
    return None

state_path = find_trainer_state(ADAPTER)
if state_path:
    with open(state_path) as f:
        history = json.load(f).get("log_history", [])
    with open(os.path.join(OUT_DIR, "loss_history.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["step", "train_loss", "eval_loss"])
        for h in history:
            if "loss" in h or "eval_loss" in h:
                w.writerow([h.get("step", ""), h.get("loss", ""), h.get("eval_loss", "")])
    print("Saved loss_history.csv")
else:
    print("No trainer_state.json found - skipping loss_history.csv "
          "(a 10-step run may not have written one).")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Tip: open metrics_summary.csv in Excel and Insert > Bar Chart for your visual.")

# """
# evaluate.py — Evaluate the fine-tuned parts-recommendation LoRA adapter.

# What it does:
#   1. Loads the base model + your saved LoRA adapter.
#   2. Runs the model over every row in val.csv.
#   3. Parses each generated "Recommended parts: ..." line into a SET of parts.
#   4. Compares that set to the TRUE set of parts for that row.
#   5. Computes precision / recall / F1 (per-example, averaged) + exact-match accuracy.
#   6. Saves two charts as PNG:
#         - metrics_bar.png   (precision / recall / F1 / exact-match)
#         - loss_curve.png    (train loss + eval loss over training, if available)
# """

# import os
# import json
# import re

# os.environ["HF_ENDPOINT"] = "https://infyartifactory.jfrog.io/artifactory/api/huggingfaceml/huggingface-remote"
# hf_token = os.environ["HF_TOKEN"]

# from util import paths
# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# import matplotlib
# matplotlib.use("Agg") # No display
# import matplotlib.pyplot as plt

# # ── Config ──────────────────────────────────────────────────────────────────
# BASE     = "HuggingFaceTB/SmolLM2-360M-Instruct"  
# ADAPTER  = os.path.join(paths.SRC, "output", "smollm3-finetuned")
# VAL_CSV  = os.path.join(paths.PROC_DATA, "val.csv")
# OUT_DIR  = os.path.join(paths.SRC, "output", "eval")
# MAX_NEW_TOKENS = 300
# LIMIT    = None    # set to e.g. 20 for a quick test run; None = all rows

# SYSTEM_PROMPT = (
#     "You are a parts recommendation service for a manufacturing dealer "
#     "parts management system. Based on the item requested to be ordered "
#     "in addition to their model, type, and reported symptom, recommend other "
#     "parts that may be in association with that requested item."
# )

# os.makedirs(OUT_DIR, exist_ok=True)

# # ── Same user-message builder as training (keep in sync!) ────────────────────
# def build_user_msg(row):
#     return (
#         f"Machine: {row['machine_model']}, "
#         f"Age: {row['machine_age_years']} years, "
#         f"Operating hours: {row['operating_hours']}, "
#         f"Subsystem: {row['subsystem']}, "
#         f"Environment: {row['environment']}. "
#         f"Symptom: {row['symptom_text']}"
#     )

# def true_parts_set(row):
#     """The gold-label parts for a row, as a normalized set."""
#     raw = str(row["label_all_part_types"])
#     return normalize_parts(raw.replace("|", ","))

# def normalize_parts(text):
#     """Turn a comma-separated parts string into a clean set of lowercase names."""
#     # strip a leading "Recommended parts:" if present
#     text = re.sub(r"(?i)recommended parts\s*:", "", text)
#     # take only the first line (models sometimes ramble afterwards)
#     text = text.strip().splitlines()[0] if text.strip() else ""
#     items = [p.strip().lower() for p in text.split(",")]
#     return {p for p in items if p}   # drop empties

# # ── Metrics ──────────────────────────────────────────────────────────────────
# def prf1(pred: set, gold: set):
#     if not pred and not gold:
#         return 1.0, 1.0, 1.0
#     if not pred or not gold:
#         return 0.0, 0.0, 0.0
#     inter = len(pred & gold)
#     precision = inter / len(pred)
#     recall    = inter / len(gold)
#     f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
#     return precision, recall, f1

# # ── Load model ───────────────────────────────────────────────────────────────
# print("Loading base model + adapter...")
# tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
# model = AutoModelForCausalLM.from_pretrained(BASE, token=hf_token)
# model = PeftModel.from_pretrained(model, ADAPTER)
# model.eval()
# print("Loading base model + adapter... DONE\n")

# # ── Run over the validation set ──────────────────────────────────────────────
# df = pd.read_csv(VAL_CSV)
# if LIMIT:
#     df = df.head(LIMIT)

# precisions, recalls, f1s, exact = [], [], [], []
# rows_out = []

# for i, (_, row) in enumerate(df.iterrows()):
#     user_msg = build_user_msg(row)
#     text = tokenizer.apply_chat_template(
#         [{"role": "system", "content": SYSTEM_PROMPT},
#          {"role": "user",   "content": user_msg}],
#         tokenize=False, add_generation_prompt=True,
#     )
#     inputs = tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
#                              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
#     generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

#     pred = normalize_parts(generated)
#     gold = true_parts_set(row)
#     p, r, f = prf1(pred, gold)
#     precisions.append(p); recalls.append(r); f1s.append(f)
#     exact.append(1.0 if pred == gold else 0.0)

#     rows_out.append({"pred": ", ".join(sorted(pred)),
#                      "gold": ", ".join(sorted(gold)),
#                      "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})
#     print(f"[{i+1}/{len(df)}]  P={p:.2f} R={r:.2f} F1={f:.2f}")

# # ── Aggregate ────────────────────────────────────────────────────────────────
# def avg(xs): return sum(xs) / len(xs) if xs else 0.0
# macro_p, macro_r, macro_f, acc = avg(precisions), avg(recalls), avg(f1s), avg(exact)

# print("\n==================  RESULTS  ==================")
# print(f"Examples evaluated : {len(df)}")
# print(f"Precision (avg)    : {macro_p:.3f}")
# print(f"Recall    (avg)    : {macro_r:.3f}")
# print(f"F1        (avg)    : {macro_f:.3f}")
# print(f"Exact-match acc    : {acc:.3f}")
# print("===============================================")

# # save per-example results for inspection
# pd.DataFrame(rows_out).to_csv(os.path.join(OUT_DIR, "per_example_results.csv"), index=False)

# # ── Chart 1: metrics bar chart ───────────────────────────────────────────────
# labels = ["Precision", "Recall", "F1", "Exact match"]
# values = [macro_p, macro_r, macro_f, acc]
# plt.figure(figsize=(6, 4))
# bars = plt.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
# plt.ylim(0, 1)
# plt.ylabel("Score")
# plt.title(f"Parts recommender — validation metrics (n={len(df)})")
# for b, v in zip(bars, values):
#     plt.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center")
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "metrics_bar.png"), dpi=150)
# print("Saved metrics_bar.png")

# # ── Chart 2: training / validation loss curve (from trainer_state.json) ───────
# def find_trainer_state(adapter_dir):
#     """trainer_state.json may be in the adapter dir or a checkpoint-* subfolder."""
#     direct = os.path.join(adapter_dir, "trainer_state.json")
#     if os.path.exists(direct):
#         return direct
#     for name in sorted(os.listdir(adapter_dir)):
#         cand = os.path.join(adapter_dir, name, "trainer_state.json")
#         if os.path.exists(cand):
#             return cand
#     return None

# state_path = find_trainer_state(ADAPTER)
# if state_path:
#     with open(state_path) as f:
#         history = json.load(f).get("log_history", [])
#     train_steps  = [h["step"] for h in history if "loss" in h]
#     train_loss   = [h["loss"] for h in history if "loss" in h]
#     eval_steps   = [h["step"] for h in history if "eval_loss" in h]
#     eval_loss    = [h["eval_loss"] for h in history if "eval_loss" in h]

#     plt.figure(figsize=(6, 4))
#     if train_loss: plt.plot(train_steps, train_loss, marker="o", label="Train loss")
#     if eval_loss:  plt.plot(eval_steps,  eval_loss,  marker="s", label="Validation loss")
#     plt.xlabel("Training step")
#     plt.ylabel("Loss")
#     plt.title("Training vs validation loss")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
#     print("Saved loss_curve.png")
# else:
#     print("No trainer_state.json found — skipping loss curve. "
#           "(It's written next to the adapter when Trainer saves; a 10-step run may not have one.)")

# print(f"\nAll outputs saved to: {OUT_DIR}")