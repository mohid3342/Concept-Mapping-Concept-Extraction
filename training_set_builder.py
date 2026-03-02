import os
import json

JSON_DIR = "json_output"
LABELED_DIR = "labeled_plus_fd"
OUTPUT_FILE = "concept_extraction_training.jsonl"

instruction_text = (
    "Extract all core computer science concepts from the following lecture JSON. "
    "Return only a newline-separated list of unique concepts. "
    "Do not explain anything."
)

def extract_concepts_from_jsonl(jsonl_path):
    concepts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            labels = data.get("label", [])

            for start, end, label_type in labels:
                if label_type != "Concept":
                    continue

                concept_text = text[start:end]
                concept_text = concept_text.replace("\n", " ").strip()

                if concept_text:
                    concepts.append(concept_text)

    # remove duplicates while preserving order
    seen = set()
    unique = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for course in os.listdir(JSON_DIR):
        course_path = os.path.join(JSON_DIR, course)
        if not os.path.isdir(course_path):
            continue

        for lecture_file in os.listdir(course_path):
            if not lecture_file.endswith(".json"):
                continue

            lec_name = lecture_file.replace(".json", "")
            json_path = os.path.join(course_path, lecture_file)

            labeled_path = os.path.join(
                LABELED_DIR, course, f"{lec_name}.jsonl"
            )

            if not os.path.exists(labeled_path):
                print(f"Skipping {course}/{lec_name} (no labeled file)")
                continue

            # Load powerpoint JSON
            with open(json_path, "r", encoding="utf-8") as jf:
                lecture_json = json.load(jf)

            # Extract concepts
            concepts = extract_concepts_from_jsonl(labeled_path)

            if not concepts:
                print(f"Skipping {course}/{lec_name} (no concepts found)")
                continue

            training_example = {
                "instruction": instruction_text,
                "input": lecture_json,
                "output": "\n".join(concepts)
            }

            out.write(json.dumps(training_example, ensure_ascii=False) + "\n")

            print(f"Added: {course}/{lec_name}")

print("Training file complete.")