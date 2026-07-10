import sys; print("RUNNING UNDER:", sys.executable)
import os

# Set up artifactory & token access for HuggingFace
os.environ["HF_ENDPOINT"] = "https://infyartifactory.jfrog.io/artifactory/api/huggingfaceml/huggingface-remote"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "86400"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "86400"
hf_token = os.environ["HF_TOKEN"]

from transformers import AutoModelForCausalLM,  AutoTokenizer
from peft import PeftModel # Parameter Efficient Fine-Tuning
from transformers import pipeline

device = "cpu"

# Models
smollm3 = "HuggingFaceTB/SmolLM3-3B"
llama = "meta-llama/Llama-3.2-1B"
gemma = "google/gemma-4-26B-A4B-it"

# Chosen Model
MODEL = smollm3

tokenizer = AutoTokenizer.from_pretrained(MODEL, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(MODEL, token=hf_token).to(device)

# Model Input
SYSTEM_PROMPT = "" \
"You are a parts recommendation service for a manufacturing dealer " \
"parts management system. Based on the item requested to be ordered " \
"in addition to their model, type, and reported symptom, recommend other " \
"parts that may be in association with that requested item."

prompt = "I need a coolant pump."
messages_think = [
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages_think,
    tokenize = False,
    add_generation_prompt = True,
    )
model_inputs = tokenizer([text], return_tensors = "pt").to(model.device)

# Generate output
generated_ids = model.generate(**model_inputs, max_new_tokens=200)

# Get & decode output
ouput_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(ouput_ids, skip_special_tokens=True))


