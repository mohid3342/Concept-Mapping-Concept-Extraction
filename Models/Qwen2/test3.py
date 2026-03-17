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
# 4. Lecture Data
# -----------------------
lecture_data = [
    {"slide_number": 1, "title": "CS 0007: Introduction to Java\u000bLecture 0", "text_blocks": [[{"text": "CS 0007: Introduction to Java\u000bLecture 0", "runs": [{"text": "CS 0007: Introduction to ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Java", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Lecture 0", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Nathan Ong", "runs": [{"text": "Nathan ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Ong", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "University of Pittsburgh", "runs": [{"text": "University of Pittsburgh", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 2, "title": "Introductions", "text_blocks": [[{"text": "Introductions", "runs": [{"text": "Introductions", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Name", "runs": [{"text": "Name", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Year", "runs": [{"text": "Year", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Why this course?", "runs": [{"text": "Why this course?", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Random fact", "runs": [{"text": "Random fact", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 3, "title": "A Note for CS and Other\u000bComputing Majors", "text_blocks": [[{"text": "A Note for CS and Other\u000bComputing Majors", "runs": [{"text": "A Note for CS and Other", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Computing Majors", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "CS 0401 (Intermediate Programming in Java) is more likely to be suitable for you.  This course is geared towards non-majors.  Talk to your academic advisor.", "runs": [{"text": "CS 0401 (Intermediate Programming in Java) is more likely to be suitable for you.  This course is geared towards non-majors.  Talk to your ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "academic advisor.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 4, "title": "Syllabus", "text_blocks": [[{"text": "Syllabus", "runs": [{"text": "Syllabus", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Can be found on Courseweb", "runs": [{"text": "Can be found on ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Courseweb", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 5, "title": "Teaching Style", "text_blocks": [[{"text": "Teaching Style", "runs": [{"text": "Teaching Style", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Because this course is geared towards non-majors, the course will be taught in a human-language inspired manner.", "runs": [{"text": "Because this course is geared towards non-majors, the course will be taught in a human-language inspired manner.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "However, students should recognize that human languages are fluid and bendable.  Programming languages (like Java) are not.", "runs": [{"text": "However, students should recognize that human languages are fluid and bendable.  Programming languages (like Java) are not.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 6, "title": "The Programming Mindset", "text_blocks": [[{"text": "The Programming Mindset", "runs": [{"text": "The Programming Mindset", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Imagine talking to your (possibly imaginary) non-cooperative little brother who claims to be following directions, but taking them as literally as possible.", "runs": [{"text": "Imagine talking to your (possibly imaginary) non-cooperative little brother who claims to be following directions, but taking them as literally as possible.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "", "runs": []}, {"text": "This is what it is like programming a computer.", "runs": [{"text": "This is what it is like programming a computer.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 7, "title": "Regarding Collaboration", "text_blocks": [[{"text": "Regarding Collaboration", "runs": [{"text": "Regarding Collaboration", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "I encourage all of you to collaborate and learn from each other by doing the homework.", "runs": [{"text": "I encourage all of you to collaborate and learn from each other by doing the homework.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "", "runs": []}, {"text": "The only requirement: please write at the top of your homework who you collaborated with.  Leaving this information out while continuing to collaborate constitutes cheating and is NOT PERMITTED.", "runs": [{"text": "The only requirement: please write at the top of your homework who you collaborated with.  Leaving this information out while continuing to collaborate constitutes cheating and is NOT PERMITTED.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": [], "notes": ""}, {"slide_number": 8, "title": "Regarding Plagiarism", "text_blocks": [[{"text": "Regarding Plagiarism", "runs": [{"text": "Regarding Plagiarism", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Plagiarism, which includes copying or using code from any source without my approval, is strictly prohibited.", "runs": [{"text": "Plagiarism, which includes copying or using code from any source without my approval, is strictly prohibited.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "", "runs": []}, {"text": "First offense: A zero for the graded item", "runs": [{"text": "First offense: A zero for the graded item", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Second offense: A zero for the course", "runs": [{"text": "Second offense: A zero for the course", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "", "runs": []}]], "images": []}, {"slide_number": 9, "title": "Academic Integrity", "text_blocks": [[{"text": "Academic Integrity", "runs": [{"text": "Academic Integrity", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "The University’s official Academic Integrity Policy can be found here: http://www.as.pitt.edu/fac/policies/academic-integrity", "runs": [{"text": "The University’s official Academic Integrity Policy can be found here: ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "http://www.as.pitt.edu/fac/policies/academic-integrity", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "", "runs": []}]], "images": []}, {"slide_number": 10, "title": "Header", "text_blocks": [[{"text": "Header", "runs": [{"text": "Header", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "For those of you that don't check your e-mail on a daily basis, you should start.", "runs": [{"text": "For those of you that don't check your e-mail on a daily basis, you should start.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Questions should be asked as soon as you have one.", "runs": [{"text": "Questions should be asked as soon as you have one.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 11, "title": "Questions?", "text_blocks": [[{"text": "Questions?", "runs": [{"text": "Questions?", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "", "runs": []}]], "images": []}

]

# -----------------------
# 5. Instruction
# -----------------------
instruction = (
    "Extract all core computer science concepts from the following lecture JSON. "
    "Return only a newline-separated list of unique concepts. Do not explain anything."
)

# -----------------------
# 6. Chunk Lecture (prevents context overflow)
# -----------------------
CHUNK_SIZE = 4
chunks = [lecture_data[i:i + CHUNK_SIZE] for i in range(0, len(lecture_data), CHUNK_SIZE)]

all_concepts = set()

# -----------------------
# 7. Run Model on Each Chunk
# -----------------------
for chunk in chunks:

    lecture_json = json.dumps(chunk, separators=(",", ":"))

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
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    concepts_output = result.split("### Output:")[-1].strip()

    for concept in concepts_output.split("\n"):
        concept = concept.strip()
        if concept:
            all_concepts.add(concept)

# -----------------------
# 8. Final Output
# -----------------------
print("\n=== FINAL CONCEPT LIST ===\n")

for concept in sorted(all_concepts):
    print(concept)
