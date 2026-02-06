from datasets import load_dataset

dataset = load_dataset("json", data_files="training_dataset.jsonl")
print(dataset)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Tokenize dataset
def tokenize(example):
    input_text = example["instruction"] + "\n" + example.get("input", "")
    target_text = example["output"]
    full_text = input_text + "\n" + target_text
    return tokenizer(full_text, truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./llama2-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  # or bf16 if supported
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()
model.save_pretrained("./llama2-finetuned")
tokenizer.save_pretrained("./llama2-finetuned")
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./llama2-finetuned")
model = AutoModelForCausalLM.from_pretrained("./llama2-finetuned", device_map="auto")
