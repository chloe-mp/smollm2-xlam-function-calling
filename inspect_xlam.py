from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch, json

mid = "Chloemp/smollm2-135m-xlam-fullft"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32)

SYSTEM_PROMPT = "You are a function-calling assistant. Available tools:\n{tools}"
ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
example = ds[0]

# ce système doit être IDENTIQUE à celui de train_xlam_job.py

messages = [
    {"role": "system", "content": SYSTEM_PROMPT.format(tools=example["tools"])},
    {"role": "user", "content": example["query"]},
]
    
inputs = tok.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
)

with torch.no_grad():
    out = model.generate(
        **inputs,                       # déplie input_ids ET attention_mask
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        # M7 — erreur de cardinalité : bon jeu de fonctions mais mauvais compte
    )

prompt_len = inputs["input_ids"].shape[1]
print("PRÉDICTION :", tok.decode(out[0, prompt_len:], skip_special_tokens=True))
print("ATTENDU    :", example["answers"])