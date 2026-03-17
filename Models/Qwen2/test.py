# test.py
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------
# 1. Load Base Model
# -----------------------
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",
    device_map="auto",
    torch_dtype=torch.float16,
)
base_model.eval()

# -----------------------
# 2. Load LoRA Adapter
# -----------------------
model = PeftModel.from_pretrained(
    base_model,
    "./qwen2-finetuned",
)
model.eval()

# -----------------------
# 3. Tokenizer Setup
# -----------------------
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------
# 4. Lecture Data (slides 4-6 only)
# -----------------------
lecture_data = [ 
    # Slide 4
    {"slide_number": 4, "title": "How a Computer is Organized", "text_blocks": [[{"text": "How a Computer is Organized"}], [{"text": "Source: https://www.doc.ic.ac.uk/~eedwards/compsys/overall.gif"}]], "images": []},
    # Slide 5
    {"slide_number": 5, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing"}], [{"text": "Source: https://upload.wikimedia.org/wikipedia/commons/e/ea/Boulier1.JPG"}]], "images": []},
    # Slide 6
    {"slide_number": 6, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing"}], [{"text": "??? \t\t– Math invented"}, {"text": "~2400 BC \t– Abacus probably invented in Babylon"}, {"text": "???\t\t– More math invented"}, {"text": "1600s\t– John Napier discovers logarithms and fast log computation"}, {"text": "1786\t\t– Johann Müller theorizes a \"Difference Engine\""}, {"text": "1822\t\t– Charles Babbage (\"father of the computer\") secures funding to build a Difference Engine (never built)"}]], "images": [], "notes": "\"difference machine\" – essentially a mechanical calculator of logarithms and trigonometric functions to approximate polynomials"}
]

# -----------------------
# 5. Prepare Prompt
# -----------------------
instruction = (
    "Extract all core computer science concepts from the following lecture JSON. "
    "Return only a newline-separated list of unique concepts. Do not explain anything."
)

prompt_dict = {
    "instruction": instruction,
    "input": lecture_data,
    "output": ""
}

lecture_json = json.dumps(prompt_dict["input"], separators=(",", ":"))

prompt = f"""### Instruction:
{instruction}

### BEGIN_LECTURE_JSON
{lecture_json}
### END_LECTURE_JSON

### Output:
"""

# -----------------------
# 6. Tokenize Input
# -----------------------
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048,
).to(model.device)

# -----------------------
# 7. Generate Concepts
# -----------------------
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.2,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

# -----------------------
# 8. Decode Output
# -----------------------
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract only the model's generated concepts
concepts_output = result.split("### Output:")[-1].strip()

print("\n=== MODEL OUTPUT ===\n")
print(concepts_output)
