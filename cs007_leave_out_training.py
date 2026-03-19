import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig


training_data = "/home/mohid/Desktop/Models/Qwen2/train_without_cs0007.jsonl"
testing_data = "/home/mohid/Desktop/Models/Qwen2/test_cs0007.jsonl"
test_labels = "/home/mohid/Desktop/Models/Qwen2/concepts_cs0007.txt"


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
    output_dir="./qwen2-finetuned_cs007",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=10,
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
trainer.save_model("./qwen2-finetuned_cs007")

# -----------------------
# Load Test Data from JSONL
# -----------------------
print("Loading test data from JSONL...")
test_examples = []
with open(testing_data, 'r') as f:
    for line in f:
        test_examples.append(json.loads(line))

#print(f"Loaded {len(test_examples)} test examples\n")



# -----------------------
# Helper Functions
# -----------------------
def load_ground_truth(file_path):
    """Load ground truth concepts from txt file"""
    concepts = set()
    with open(file_path, 'r') as f:
        for line in f:
            concept = line.strip().lower()
            if concept:
                concepts.add(concept)
    return concepts

def normalize_concept(concept):
    """Normalize concept text for comparison"""
    concept = concept.strip().lower()
    concept = concept.lstrip('•-*0123456789. ')
    return concept

def calculate_f1(predicted, ground_truth):
    """Calculate precision, recall, and F1 score"""
    pred_normalized = {normalize_concept(c) for c in predicted}
    gt_normalized = {normalize_concept(c) for c in ground_truth}
    
    true_positives = len(pred_normalized & gt_normalized)
    false_positives = len(pred_normalized - gt_normalized)
    false_negatives = len(gt_normalized - pred_normalized)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'predicted_count': len(pred_normalized),
        'ground_truth_count': len(gt_normalized)
    }

# -----------------------
# Load Ground Truth
# -----------------------
print("Loading ground truth concepts...")
ground_truth_concepts = load_ground_truth(test_labels)
print(f"Loaded {len(ground_truth_concepts)} ground truth concepts\n")


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Running on: {device}")

# -----------------------

# 2. Load Base Model

# -----------------------

base_model = AutoModelForCausalLM.from_pretrained(
"Qwen/Qwen2-1.5B",
torch_dtype=dtype,
device_map="auto"
)

# -----------------------

# 3. Load LoRA Adapter

# -----------------------

model_test = PeftModel.from_pretrained(
base_model,
"./qwen2-finetuned"
)

# -----------------------
# Run Model on Test Data
# -----------------------
print("Extracting concepts from test data...\n")

all_concepts = set()

for idx, example in enumerate(test_examples):
    print(f"Processing test example {idx + 1}/{len(test_examples)}...")
    
    # Extract the input data (lecture slides)
    lecture_json = json.dumps(example["input"], separators=(",", ":"))
    instruction = example["instruction"]

    prompt = f"""### Instruction:
{instruction}

### BEGIN_LECTURE_JSON
{lecture_json}
### END_LECTURE_JSON

### Output:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model_test.device)

    with torch.no_grad():
        outputs = model_test.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip()

    # Extract concepts from response
    for concept in response.split("\n"):
        concept = concept.strip()
        if concept:
            all_concepts.add(concept)

# -----------------------
# Calculate Metrics
# -----------------------
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

metrics = calculate_f1(all_concepts, ground_truth_concepts)

print(f"\nPredicted Concepts: {metrics['predicted_count']}")
print(f"Ground Truth Concepts: {metrics['ground_truth_count']}")
print(f"\nTrue Positives: {metrics['true_positives']}")
print(f"False Positives: {metrics['false_positives']}")
print(f"False Negatives: {metrics['false_negatives']}")
print(f"\n{'Precision:':<15} {metrics['precision']:.4f}")
print(f"{'Recall:':<15} {metrics['recall']:.4f}")
print(f"{'F1 Score:':<15} {metrics['f1']:.4f}")

# -----------------------
# Detailed Comparison
# -----------------------
print("\n" + "="*60)
print("PREDICTED CONCEPTS")
print("="*60)
for concept in sorted(all_concepts):
    print(concept)

print("\n" + "="*60)
print("MATCHED CONCEPTS (True Positives)")
print("="*60)
pred_normalized = {normalize_concept(c) for c in all_concepts}
gt_normalized = {normalize_concept(c) for c in ground_truth_concepts}
matched = pred_normalized & gt_normalized
for concept in sorted(matched):
    print(concept)

print("\n" + "="*60)
print("MISSED CONCEPTS (False Negatives)")
print("="*60)
missed = gt_normalized - pred_normalized
for concept in sorted(missed):
    print(concept)

print("\n" + "="*60)
print("EXTRA CONCEPTS (False Positives)")
print("="*60)
extra = pred_normalized - gt_normalized
for concept in sorted(extra):
    print(concept)

# -----------------------
# Save Results
# -----------------------
results = {
    'metrics': metrics,
    'predicted_concepts': sorted(list(all_concepts)),
    'ground_truth_concepts': sorted(list(ground_truth_concepts)),
    'matched_concepts': sorted(list(matched)),
    'missed_concepts': sorted(list(missed)),
    'extra_concepts': sorted(list(extra))
}

with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Results saved to 'evaluation_results.json'")
print("="*60)