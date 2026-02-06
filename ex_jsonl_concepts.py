import os
import json

INPUT_DIR = "jsonl_files"
OUTPUT_DIR = "jsonl_files_text"  # change if you want a separate folder

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".jsonl"):
        continue

    filepath = os.path.join(INPUT_DIR, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            text = data["text"]
            labels = data.get("label", [])
            course = data.get("course", "unknown_course")
            lec = data.get("lec", "unknown_lec")

            concepts = []

            for start, end, label_type in labels:
                if label_type != "Concept":
                    continue

                # slice using character indices
                concept_text = text[start:end]

                # clean up whitespace
                concept_text = concept_text.replace("\n", " ").strip()

                if concept_text:
                    concepts.append(concept_text)

            # remove duplicates while preserving order
            seen = set()
            unique_concepts = []
            for c in concepts:
                if c not in seen:
                    seen.add(c)
                    unique_concepts.append(c)

            # write output file
            output_filename = f"{course}_{lec}.txt"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, "w", encoding="utf-8") as out:
                for concept in unique_concepts:
                    out.write(concept + "\n")

            print(f"Written: {output_filename}")
