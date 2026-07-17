import sys; print("RUNNING UNDER:", sys.executable)
from util import paths
import os

# Set up artifactory & token access for HuggingFace
os.environ["HF_ENDPOINT"] = "https://infyartifactory.jfrog.io/artifactory/api/huggingfaceml/huggingface-remote"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "86400"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "86400"
hf_token = os.environ["HF_TOKEN"]
# os.environ["HF_HUB_OFFLINE"] = "1" # Use if model has already been downloaded into the cache

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# ── Path ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(paths.SRC, "output", "smollm3-finetuned")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL =  "HuggingFaceTB/SmolLM2-360M-Instruct" # "HuggingFaceTB/SmolLM3-3B"
MAX_LENGTH = 512

SYSTEM_PROMPT = (
    "You are a parts recommendation service for a manufacturing dealer "
    "parts management system. Based on the item requested to be ordered "
    "in addition to their model, type, and reported symptom, recommend other "
    "parts that may be in association with that requested item."
)

# ── Tokenizer & Model ──────────────────────────────────────────────────────────
print("Loading tokenizer and model...")
import time
t = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"tokenizer step took {time.time()-t:.1f}s" + "\n")

t = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, token=hf_token)
print(f"model step took {time.time()-t:.1f}s")
print("Loading tokenizer and model... DONE" + "\n")

# ── LoRA ───────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Dataset ────────────────────────────────────────────────────────────────────
def build_messages(row):
    user_content = (
        f"Machine: {row['machine_model']}, "
        f"Age: {row['machine_age_years']} years, "
        f"Operating hours: {row['operating_hours']}, "
        f"Subsystem: {row['subsystem']}, "
        f"Environment: {row['environment']}. "
        f"Symptom: {row['symptom_text']}"
    )
    parts = row["label_all_part_types"].replace("|", ", ")
    assistant_content = f"Recommended parts: {parts}"
    return user_content, assistant_content


class PartsDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.examples = []

        for _, row in df.iterrows():
            user_msg, assistant_msg = build_messages(row)

            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            full_enc   = tokenizer(full_text,   truncation=True, max_length=MAX_LENGTH)
            prompt_enc = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH)

            input_ids      = torch.tensor(full_enc["input_ids"])
            attention_mask = torch.tensor(full_enc["attention_mask"])
            labels         = input_ids.clone()
            prompt_len     = len(prompt_enc["input_ids"])
            labels[:prompt_len] = -100  # only learn to predict the assistant reply

            self.examples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    return {
        "input_ids":      pad_sequence([b["input_ids"]      for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
        "labels":         pad_sequence([b["labels"]         for b in batch], batch_first=True, padding_value=-100),
    }


print("Building datasets...")
train_dataset = PartsDataset(os.path.join(paths.PROC_DATA, "train.csv"))
val_dataset   = PartsDataset(os.path.join(paths.PROC_DATA, "val.csv"))
print(f"Train: {len(train_dataset)} examples | Val: {len(val_dataset)} examples" + "\n")

print("Checking train dataset:")
print(tokenizer.decode(train_dataset[0]["input_ids"]) + "\n")

# ── Training ───────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # effective batch size = 8
    eval_strategy="epoch",
    save_strategy="epoch",
    max_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    fp16=False,   # CPU does not support fp16 training
    bf16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

print("Starting training...")
trainer.train()

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter and tokenizer saved to: {OUTPUT_DIR}")
