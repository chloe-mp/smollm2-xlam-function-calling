# /// script
# dependencies = [
#     "trl>=1.10",
#     "transformers",
#     "datasets",
#     "torch",
#     "peft",
#     "bitsandbytes",
#     "hf_transfer",
#     "trackio",
# ]
# ///
"""
Fine-tuning LoRA / QLoRA de SmolLM2-1.7B sur xLAM function calling, pour HF Jobs.

Même données, même split, même prompt, même template que le full FT du 135M
(train_xlam_job.py). Seuls changent : la taille du modèle et la méthode.

Lancement (un job par méthode) :
    hf jobs uv run --flavor a100-large --timeout 3h --secrets HF_TOKEN \
      train_xlam_peft_job.py --method lora
    hf jobs uv run --flavor a100-large --timeout 3h --secrets HF_TOKEN \
      train_xlam_peft_job.py --method qlora

LoRA vs QLoRA ne diffèrent QUE par la quantification 4 bits du modèle de base :
même rang, mêmes modules, même LR. Sinon la comparaison ne mesure plus la
quantification mais un mélange d'hyperparamètres.
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

# --- Constantes d'expérience -------------------------------------------------
# SEED, TEST_SIZE et SYSTEM_PROMPT : STRICTEMENT identiques à train_xlam_job.py
# et eval_xlam_job.py, sinon le split held-out diffère (contamination) ou le
# prompt d'éval ne correspond plus à celui de l'entraînement.
SEED = 42
TEST_SIZE = 0.05
EVAL_SUBSET = 500  # sous-échantillon pour le suivi pendant l'entraînement

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
# Template byte-identique à celui du 135M-Instruct (vérifié), et aucun token à
# ajouter : <|im_start|>/<|im_end|> existent déjà dans le vocab du base 1.7B.
# Donc pas d'embeddings neufs à entraîner, LoRA suffit.
TEMPLATE_SOURCE = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
HUB_MODEL_ID = "Chloemp/smollm2-1.7b-xlam-{method}"

SYSTEM_PROMPT = "You are a function-calling assistant. Available tools:\n{tools}"

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def to_messages(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=example["tools"])},
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["answers"]},
        ]
    }


class TrackioPeftConfigFix(TrainerCallback):
    """
    Contourne un crash trackio au premier push (step 500, save_steps).

    Le TrackioCallback de transformers logge `model.peft_config` tel quel, en
    dict imbriqué. Au push, trackio exporte la config en Parquet ; les champs
    vides du LoraConfig (`rank_pattern={}`, `alpha_pattern={}`, `loftq_config={}`)
    deviennent des structs sans champ, que pyarrow refuse d'écrire -> exception
    -> train() s'arrête. Le full FT du 135M n'avait pas de peft_config, d'où
    l'absence du bug. Reproduit en local (trackio 0.38.1, pyarrow 25).

    Correctif : juste après le setup de trackio (les callbacks passés à
    SFTTrainer s'exécutent APRÈS ceux de report_to), on écrase la clé par une
    chaîne JSON. La config LoRA reste lisible dans trackio.
    """

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or "trackio" not in args.report_to:
            return
        import trackio

        peft_config = getattr(model, "peft_config", None) or {}
        as_json = json.dumps({k: v.to_dict() for k, v in peft_config.items()}, default=str)
        trackio.config.update({"peft_config": as_json}, allow_val_change=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "qlora"], required=True)
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="smoke test : ex. 20 (-1 = 1 epoch)"
    )
    args = parser.parse_args()

    token = os.environ["HF_TOKEN"]
    hub_model_id = HUB_MODEL_ID.format(method=args.method)
    smoke = args.max_steps > 0

    # --- Données (identique au full FT) ----------------------------------------
    raw = load_dataset("Salesforce/xlam-function-calling-60k", token=token)["train"]
    converted = raw.map(to_messages, remove_columns=raw.column_names)
    split = converted.train_test_split(test_size=TEST_SIZE, seed=SEED)

    train_dataset = split["train"]
    eval_dataset = split["test"].select(range(EVAL_SUBSET))

    # --- Quantification (QLoRA uniquement) -----------------------------------
    # NF4 + double quantification : la recette du papier QLoRA. Les poids de
    # base sont stockés en 4 bits et déquantifiés en bf16 pour chaque calcul.
    quantization_config = None
    if args.method == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # --- Adaptateurs LoRA (identiques pour les deux méthodes) -----------------
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,  # échelle effective alpha/r = 2
        lora_dropout=0.05,
        # Toutes les couches linéaires (attention + MLP), pas seulement q/v :
        # nécessaire pour approcher le full FT sur une tâche de format strict.
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Entraînement --------------------------------------------------------
    training_args = SFTConfig(
        output_dir="./sft_output",
        num_train_epochs=1,
        max_steps=args.max_steps,
        # 8 x 2 = batch effectif 16, comme le full FT du 135M : même nombre de
        # steps (~3560) pour la même époque, donc courbes comparables.
        # 8 et pas 16 par device : les activations d'un 1.7B sont ~10x plus lourdes.
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        # LoRA : LR ~4x plus haut qu'en full FT (5e-5). Les adaptateurs partent
        # de zéro (B=0) et n'ont que ~1 % des paramètres à bouger.
        learning_rate=2e-4,
        # Le full FT n'avait pas de warmup (noté dans EVAL_TODO) ; à 2e-4 c'est
        # plus risqué, donc 3 % des steps de montée.
        # transformers 5 : warmup_ratio n'existe plus ; un float < 1 ici = ratio.
        warmup_steps=0.03,
        max_length=2048,  # max mesuré sur xLAM : 1653 tokens, aucune troncature
        bf16=True,
        # Défaut TRL = float32 : il faut forcer bf16, sinon le 1.7B pèse 6,8 Go
        # au lieu de 3,4 Go et le LoRA "non quantifié" est artificiellement lourd.
        # En QLoRA, dtype ne concerne que les couches non quantifiées (norms, embeddings).
        model_init_kwargs={"dtype": "bfloat16"},
        chat_template_path=TEMPLATE_SOURCE,
        gradient_checkpointing=True,  # défaut TRL, explicité : c'est ce qui fait tenir le batch 8
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        save_total_limit=2,
        run_name=f"{args.method}-1.7b" + ("-smoke" if smoke else ""),
        # Pas de push en smoke test : on ne veut pas de repo pollué par 20 steps.
        push_to_hub=not smoke,
        hub_model_id=hub_model_id,
        hub_private_repo=False,
        hub_strategy="every_save",  # pousse à chaque save_steps (≈8 checkpoints)
        seed=SEED,
        report_to="trackio",
    )

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        quantization_config=quantization_config,
        callbacks=[TrackioPeftConfigFix()],
    )

    # Vérifications à lire dans les logs AVANT de laisser tourner 1 h.
    trainer.model.print_trainable_parameters()
    tok = trainer.processing_class
    print(f"eos={tok.eos_token} pad={tok.pad_token} | Train : {len(train_dataset)} | Eval : {len(eval_dataset)}")
    print(f"Méthode : {args.method} | 4 bits : {getattr(trainer.model, 'is_loaded_in_4bit', False)}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Le coût EST le résultat de LoRA vs QLoRA : on le persiste avec le modèle,
    # pas seulement dans les logs (disque du job éphémère).
    stats = {
        "method": args.method,
        "model": MODEL_NAME,
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "train_runtime_s": round(elapsed),
        "gpu": torch.cuda.get_device_name(0),
        "global_steps": trainer.state.global_step,
    }
    print("RUN STATS :", json.dumps(stats))
    with open(os.path.join(training_args.output_dir, "run_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if not smoke:
        # Pousse l'ADAPTATEUR (~70 Mo), pas le modèle fusionné. Pour l'éval :
        # AutoModelForCausalLM.from_pretrained(hub_model_id) le charge si peft
        # est installé (il retrouve le base via adapter_config.json).
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
