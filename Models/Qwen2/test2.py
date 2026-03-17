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
# 4. Lecture Data (slides 10-13 only)
# -----------------------
lecture_data = [
    # Slide 10
    {"slide_number": 10, "title": "Ada Lovelace", "text_blocks": [[{"text": "Ada Lovelace"}], [{"text": "In 1842, she recognized the usage of multiple, non-direct, complex operations and memory could generate Bernoulli numbers using Babbage's Analytical Engine."}, {"text": "This is the first computer program."}]], "images": [{"alt_text": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Ada_Lovelace_portrait.jpg"}]},
    # Slide 11
    {"slide_number": 11, "title": "", "text_blocks": [[{"text": ""}], [{"text": "Side note: Yes there is a fan-fiction ship musical for Babbage and Lovelace."}]], "images": []},
    # Slide 12
    {"slide_number": 12, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing"}], [{"text": "1885\t– AT&T (the company behind the operating system Unix and the programming language C) founded"}, {"text": "1906\t– Xerox (the company behind desktops and the computer mouse) founded"}, {"text": "1907\t– Vacuum tube invented"}, {"text": "1911\t– IBM (the company behind hard disks and floppy disks) founded"}, {"text": "1931\t– Charles Wynn-Williams published first usage of electronics in computation"}]], "images": []},
    # Slide 13
    {"slide_number": 13, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing"}], [{"text": "1938\t– Claude Shannon published first usage of electronics in Boolean algebra computation"}, {"text": "1946\t– Alan Turing publishes a paper on programs stored and read from tape"}, {"text": "– The ENIAC, the first electronic general-purpose computer, was developed"}, {"text": "1947 – Transistor invented"}, {"text": "– Cathode Ray Tube (CRT) Random Access Memory (RAM) invented"}]], "images": []},
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
