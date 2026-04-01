import torch
import json
import re

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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------
# Load Test Data from JSONL
# -----------------------
print("Loading test data from JSONL...")
test_examples = []
with open(testing_data, 'r') as f:
    for line in f:
        test_examples.append(json.loads(line))

print(f"Loaded {len(test_examples)} test examples\n")



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
device_map= None
)

# -----------------------

# 3. Load LoRA Adapter

# -----------------------

model_test = PeftModel.from_pretrained(
base_model,
"./qwen2-finetuned3_cs007"
)

# -----------------------
# Run Model on Test Data
# -----------------------
print("Extracting concepts from test data...\n")

all_concepts = set()

for idx, example in enumerate(test_examples):
    if idx % 5 == 0:
        print(f"Processing test example {idx + 1}/{len(test_examples)}...")
    
    # Extract the input data (lecture slides)
    # Parse the input string to get the actual data
    lecture_data = json.loads(example["input"])
    lecture_json = json.dumps(lecture_data, separators=(",", ":"))
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
            temperature=0.3,  # Lower temperature for more consistent concept extraction
            do_sample=True,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip()

    # Debug: show raw output for every example
    print(f"\n[Example {idx + 1}/{len(test_examples)}]")
    print(f"Raw response: {repr(response[:300])}")

    # Parse model output into concepts (newline, comma, semicolon, pipe delimiters)
    concepts = []
    filtered_concepts = []
    if response.lower() != "none":
        raw_concepts = [c.strip() for c in re.split(r"[\n,;|]", response) if c.strip() and c.strip().lower() != 'none']

        for c in raw_concepts:
            clean = c.strip()
            if not clean:
                continue
            if len(clean) > 40:
                # Skip overly long concepts
                continue
            filtered_concepts.append(clean)
            concepts.append(clean)

    print(f"Raw split concepts: {raw_concepts if 'raw_concepts' in locals() else []}")
    print(f"Filtered concepts (<=40 chars): {filtered_concepts}")
    print(f"Parsed concepts ({len(concepts)}): {concepts}")
    
    # Add all valid concepts
    for concept in concepts:
        all_concepts.add(concept)
    
    print(f"all_concepts so far ({len(all_concepts)}): {all_concepts}")
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

with open('evaluation_results_cs0007.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Results saved to 'evaluation_results_cs0007.json'")
print("="*60)
