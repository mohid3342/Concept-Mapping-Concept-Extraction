import json
import os

# Files to process
files = [
    "train_without_cs0007.jsonl",
    "test_cs0007.jsonl"
]



# Text to add
extra_instruction = "\n- Do not extract code segments, only pure concepts\n"

def update_file(file_path):
    updated_lines = []

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            # Modify instruction
            if "instruction" in data:
                if "do not extract code segments" not in data["instruction"].lower():
                    data["instruction"] += extra_instruction

            updated_lines.append(json.dumps(data))

    # Create new filename
    base, ext = os.path.splitext(file_path)
    new_file = f"{base}_revised{ext}"

    # Write to new file
    with open(new_file, "w") as f:
        for line in updated_lines:
            f.write(line + "\n")

    print(f"Saved revised file: {new_file}")


# Run for all files
for file in files:
    update_file(file)








def sync_test_instruction(train_file, test_file):
    import json
    import os

    # Get instruction from first training example
    with open(train_file, "r") as f:
        first_line = f.readline()
        train_data = json.loads(first_line)
        train_instruction = train_data["instruction"]

    print("Using training instruction:\n")
    print(train_instruction)
    print("\n---\n")

    updated_lines = []

    with open(test_file, "r") as f:
        for line in f:
            data = json.loads(line)

            # Replace instruction
            data["instruction"] = train_instruction

            updated_lines.append(json.dumps(data))

    # Save new file
    base, ext = os.path.splitext(test_file)
    new_file = f"{base}_synced{ext}"

    with open(new_file, "w") as f:
        for line in updated_lines:
            f.write(line + "\n")

    print(f"Saved synced test file: {new_file}")




sync_test_instruction(
    "train_without_cs0007_revised.jsonl",
    "test_cs0007_revised.jsonl"
)


def apply_train_instruction_to_test(train_file, test_file):
    # Get the FIRST instruction from train file
    with open(train_file, "r") as f:
        first_line = json.loads(next(f))
        train_instruction = first_line.get("instruction", "")

    updated_lines = []

    # Apply that exact instruction to all test entries
    with open(test_file, "r") as f:
        for line in f:
            data = json.loads(line)
            data["instruction"] = train_instruction
            updated_lines.append(json.dumps(data))

    # Save as a new file
    base, ext = os.path.splitext(test_file)
    new_file = f"{base}_instruction_aligned{ext}"

    with open(new_file, "w") as f:
        for line in updated_lines:
            f.write(line + "\n")

    print(f"Saved new file: {new_file}")
   
apply_train_instruction_to_test("train_without_cs0007.jsonl", "test_cs0007.jsonl")

print("Done.")
