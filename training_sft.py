import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


print("Starting training...")

# -----------------------------
# Configuration
# -----------------------------
model_name = "meta-llama/Llama-2-7b-chat-hf"
data_path = "your_dataset.jsonl"
output_dir = "./llama2-7b-concept-sft"

# -----------------------------
# Load Model (4-bit QLoRA)
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Add LoRA
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset("json", data_files=data_path)["train"]

def format_example(example):
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}

dataset = dataset.map(format_example)

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,          # start with 2 (avoid overfitting)
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

# -----------------------------
# Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=2048,
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save LoRA Adapter
# -----------------------------
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete.")

