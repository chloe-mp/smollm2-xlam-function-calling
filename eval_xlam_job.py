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


import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, clone_chat_template

SEED = 42
TEST_SIZE = 0.05
EVAL_SKIP = 500

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
TEMPLATE_SOURCE = "HuggingFaceTB/SmolLM2-135M-Instruct"
HUB_MODEL_ID = "Chloemp/smollm2-135m-xlam-fullft"  

SYSTEM_PROMPT = "You are a function-calling assistant. Available tools:\n{tools}"

def to_messages(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=example["tools"])},
            {"role": "user", "content": example["query"]},
        ]
    }


def main() -> None:
    token = os.environ["HF_TOKEN"]  

    # --- Données -------------------------------------------------------------
    raw = load_dataset("Salesforce/xlam-function-calling-60k", token=token)["train"]
    converted = raw.map(to_messages)
    split = converted.train_test_split(test_size=TEST_SIZE, seed=SEED)

    train_dataset = split["train"]
    eval_dataset = split["test"].select(range(EVAL_SKIP, len(split["test"])))

    train_split = frozenset(train_dataset["query"])
    test_split = frozenset(eval_dataset["query"])
    duplicates = train_split.intersection(test_split)
    print(len(duplicates))

    

    print(len(eval_dataset))
    print(eval_dataset.column_names)

    contaminated_rows = eval_dataset.filter(lambda x: x["query"] in duplicates)
    print(contaminated_rows)

    clean_eval = eval_dataset.filter(lambda x: x["query"] not in duplicates)
    print(clean_eval)


if __name__ == "__main__":
    main()
