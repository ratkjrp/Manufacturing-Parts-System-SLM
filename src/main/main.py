import os
from util import paths
os.environ["HF_ENDPOINT"] = "https://infyartifactory.jfrog.io/artifactory/api/huggingfaceml/huggingface-remote"
hf_token = os.environ["HF_TOKEN"]

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER = os.path.join(paths.SRC, "output", "smollm3-finetuned")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, token=hf_token)
model = PeftModel.from_pretrained(model, ADAPTER)   # Applying generated LoRA adapter
model.eval()

SYSTEM_PROMPT = (
    "You are a parts recommendation service for a manufacturing dealer "
    "parts management system. Based on the item requested to be ordered "
    "in addition to their model, type, and reported symptom, recommend other "
    "parts that may be in association with that requested item."
)
user_msg = "Machine: Sedan, Age: 5 years, Symptom: grinding noise on braking"

text = tokenizer.apply_chat_template(
    [{"role":"system","content":SYSTEM_PROMPT}, {"role":"user","content":user_msg}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))