import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# -----------------------
# 1. Load Dataset
# -----------------------
dataset = load_dataset(
    "json",
    data_files="/home/mohid/Desktop/Models/Qwen2/concept_extraction_training.jsonl",
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

dataset = dataset.map(format_example)

# -----------------------
# 2. Load Model in 4-bit
# -----------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",
    quantization_config=bnb_config,
    device_map="auto",
)

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
    output_dir="./qwen2-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,     # IMPORTANT: disable AMP
    bf16=False,     # IMPORTANT: disable BF16
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none",
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
trainer.save_model("./qwen2-finetuned")
