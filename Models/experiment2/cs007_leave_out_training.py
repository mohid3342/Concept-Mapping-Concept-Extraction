import torch
import json

#TODO: Create a sequential process to train and test all jsonl

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from random import random
#from accelerate import Accelerator


training_data = "train_without_cs0007_03_25.jsonl"
testing_data = "test_cs0007_03_25.jsonl"
test_labels = "concepts_cs0007.txt"


# -----------------------
# 1. Load Dataset
# -----------------------
dataset = load_dataset(
    "json",
    data_files=training_data,
    split="train",
)

def format_example(example):
    lecture_json = json.dumps(example["input"], separators=(",", ":"))
    return {
        "text": f"""### Instruction:
{example["instruction"]}

### BEGIN_LECTURE_JSON
{lecture_json}
### END_LECTURE_JSON

### Output:
{example["output"]}"""
    }

empty_count = 0
total_count = 0

for example in dataset:
    output = example["output"]

    # Handle escaped JSON
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except:
            pass

    if isinstance(output, list) and len(output) == 0:
        empty_count += 1

    total_count += 1

print(f"Empty outputs ([]) count: {empty_count}")
print(f"Total examples: {total_count}")
print(f"Percentage empty: {empty_count / total_count:.2%}\n")




def downsample_empty(example):
    output = example["output"]

    # Parse if needed
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except:
            pass

    # Keep all non-empty
    if isinstance(output, list) and len(output) > 0:
        return True

    # Keep only some empty ones (e.g., 30%)
    return random() < 0

dataset = dataset.map(format_example)
dataset = dataset.filter(downsample_empty)

empty_count = 0
total_count = len(dataset)

for example in dataset:
    output = example["output"]

    if isinstance(output, str):
        try:
            output = json.loads(output)
        except:
            pass

    if isinstance(output, list) and len(output) == 0:
        empty_count += 1

print(f"After downsampling:")
print(f"Empty outputs: {empty_count}")
print(f"Total: {total_count}")
print(f"Percentage empty: {empty_count / total_count:.2%}\n")

# -----------------------
# 2. Load Model in 4-bit
# -----------------------
#accelerator = Accelerator()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",
    quantization_config=bnb_config,
    device_map=None
)

#model.gradient_checkpointing_enable()

model.config.use_cache = False  # Required for gradient checkpointing

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------
# 3. Apply LoRA (4GB Safe)
# -----------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()

# -----------------------
# 4. Tokenize (Limit Sequence Length)
# -----------------------
MAX_LENGTH = 192  # Safe for 4GB RTX A400

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
)

# -----------------------
# 5. Training Config (NO AMP)
# -----------------------
sft_config = SFTConfig(
    output_dir="./qwen2-finetuned3_cs007",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=False,     # IMPORTANT: disable AMP
    bf16=False,     # IMPORTANT: disable BF16
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none",
    do_eval=True
)

# -----------------------
# 6. Trainer
# -----------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=sft_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./qwen2-finetuned3_cs007")