import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# -----------------------
# 1. Load Dataset
# -----------------------
dataset = load_dataset(
    "json",
    data_files="/mnt/data/concept_extraction_training.jsonl",
    split="train",
)

# Adjust this if your JSON keys differ
def format_example(example):
    return {
        "text": f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# 3. Apply LoRA
# -----------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()

# -----------------------
# 4. Training Arguments
# -----------------------
training_args = TrainingArguments(
    output_dir="./qwen2-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none",
)

# -----------------------
# 5. Trainer
# -----------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()

trainer.save_model("./qwen2-finetuned")