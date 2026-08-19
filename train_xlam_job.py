# /// script
# dependencies = [
#     "trl>=1.10",
#     "transformers",
#     "datasets",
#     "torch",
#     "hf_transfer",
#     "trackio",
# ]
# ///
"""
Fine-tuning full de SmolLM2-135M sur xLAM function calling, pour HF Jobs.

Lancement :
    hf jobs uv run \
      --flavor a10g-small \
      --timeout 1h \
      --secrets HF_TOKEN \
      train_xlam_job.py

Les dépendances sont déclarées en tête de fichier (format PEP 723), donc
UV les installe automatiquement sur la machine distante. Pas besoin de --with.
"""

import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, clone_chat_template

# --- Constantes d'expérience -------------------------------------------------
# SEED et TEST_SIZE doivent rester identiques entre tous les runs comparés
# ET identiques à ceux du script d'évaluation, sinon le split held-out diffère
# et des exemples d'entraînement se retrouvent dans l'éval.
SEED = 42
TEST_SIZE = 0.05
EVAL_SUBSET = 500  # sous-échantillon pour le suivi pendant l'entraînement

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
TEMPLATE_SOURCE = "HuggingFaceTB/SmolLM2-135M-Instruct"
HUB_MODEL_ID = "Chloemp/smollm2-135m-xlam-fullft"  

# Doit être STRICTEMENT identique au SYSTEM_PROMPT du script d'évaluation.
SYSTEM_PROMPT = "You are a function-calling assistant. Available tools:\n{tools}"

# Accélère les téléchargements depuis le Hub sur la machine distante.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def to_messages(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=example["tools"])},
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["answers"]},
        ]
    }


def main() -> None:
    token = os.environ["HF_TOKEN"]  

    # --- Données -------------------------------------------------------------
    raw = load_dataset("Salesforce/xlam-function-calling-60k", token=token)["train"]
    converted = raw.map(to_messages, remove_columns=raw.column_names)
    split = converted.train_test_split(test_size=TEST_SIZE, seed=SEED)

    train_dataset = split["train"]
    eval_dataset = split["test"].select(range(EVAL_SUBSET))

    # --- Modèle --------------------------------------------------------------
    # Pas de .to(device) : sur GPU, le Trainer place le modèle lui-même.
    # Un .to("cpu") explicite forcerait un aller-retour inutile.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model, tokenizer, added_tokens = clone_chat_template(
        model=model,
        tokenizer=tokenizer,
        source_tokenizer_path=TEMPLATE_SOURCE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokens ajoutés : {added_tokens}")
    print(f"pad={tokenizer.pad_token} eos={tokenizer.eos_token}")
    print(f"Train : {len(train_dataset)} | Eval : {len(eval_dataset)}")

    # --- Entraînement --------------------------------------------------------
    training_args = SFTConfig(
        output_dir="./sft_output",
        num_train_epochs=1,
        # Batch 16 au lieu de 4 : un GPU est sous-employé en dessous.
        # Conséquence : ~3560 steps au lieu de ~14250 pour la même époque.
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        max_length=2048,
        # bf16 : disponible sur A10G/A100, indisponible sur MPS.
        bf16=True,
        # Intervalles recalés sur ~3560 steps (règle : ~15 évals sur le run).
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        save_total_limit=2,
        run_name="fullft-2000",
        # Sans ceci, tout disparaît à la fin du job : le disque est éphémère.
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=False,
        hub_strategy="every_save", 
        seed=SEED,
        report_to="trackio",  # suivi des courbes ; "none" pour désactiver
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()