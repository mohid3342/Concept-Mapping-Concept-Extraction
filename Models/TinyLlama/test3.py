import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig

# -----------------------

# 1. Detect Device

# -----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Running on: {device}")

# -----------------------
# 2. Load Model
# -----------------------

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./tinyllama-finetuned"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Running on:", "cuda" if torch.cuda.is_available() else "cpu")

# tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(base_model)

# load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

# load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

model.eval()

# -----------------------
# 3. Lecture Data
# -----------------------

lecture_data = [
    # Slide 12
    {"slide_number": 12, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing"}], [{"text": "1885\t– AT&T (the company behind the operating system Unix and the programming language C) founded"}, {"text": "1906\t– Xerox (the company behind desktops and the computer mouse) founded"}, {"text": "1907\t– Vacuum tube invented"}, {"text": "1911\t– IBM (the company behind hard disks and floppy disks) founded"}, {"text": "1931\t– Charles Wynn-Williams published first usage of electronics in computation"}]], "images": []}
]

# -----------------------
# 4. Instruction
# -----------------------

instruction = (
    "Extract the core computer science concepts from this lecture. "
    "Return ONLY a newline-separated list of unique concepts."
)

lecture_json = json.dumps(lecture_data, separators=(",", ":"))

prompt = f"""### Instruction:
{instruction}

### BEGIN_LECTURE_JSON
{lecture_json}
### END_LECTURE_JSON

### Output:
"""

# -----------------------
# 5. Run Model
# -----------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=1200
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== RAW MODEL OUTPUT ===\n")
print(decoded)

if "### Output:" in decoded:
    response = decoded.split("### Output:")[-1].strip()
else:
    response = decoded.strip()


# -----------------------
# 6. Print Result
# -----------------------

print("\n=== CONCEPTS ===\n")
print(response)
