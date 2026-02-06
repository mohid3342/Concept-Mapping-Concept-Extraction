import os
import re
import json
import shutil

BASE_DIR = "json_output"

def change_title_name():
    for folder_name in os.listdir(BASE_DIR):
        old_path = os.path.join(BASE_DIR, folder_name)

        if not os.path.isdir(old_path):
            continue

        # look for 4 consecutive digits (course number)
        match = re.search(r"(\d{4})", folder_name)

        if not match:
            print(f"Skipping (no course number found): {folder_name}")
            continue

        course_num = match.group(1)
        new_folder_name = f"cs{course_num}"
        new_path = os.path.join(BASE_DIR, new_folder_name)

        if folder_name == new_folder_name:
            continue

        if os.path.exists(new_path):
            print(f"Already exists, skipping rename: {new_folder_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {folder_name} → {new_folder_name}")
   

def rename_lecture_files_007(base_dir="json_output/cs0007"):
    for filename in os.listdir(base_dir):
        match = re.search(r"Lecture\s*(\d+)", filename, re.IGNORECASE)
        if not match:
            continue

        lec_num = int(match.group(1))
        new_name = f"lec{lec_num:02d}.json"

        old_path = os.path.join(base_dir, filename)
        new_path = os.path.join(base_dir, new_name)

        if os.path.exists(new_path):
            print(f"Skipping (already exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}") 

def rename_0447_lecture_files(base_dir="json_output/cs0447"):
    for filename in os.listdir(base_dir):
        if not filename.endswith(".json"):
            continue

        # match leading digits at start of filename
        match = re.match(r"(\d+)", filename)
        if not match:
            continue

        lec_num = int(match.group(1))
        new_name = f"lec{lec_num:02d}.json"

        old_path = os.path.join(base_dir, filename)
        new_path = os.path.join(base_dir, new_name)

        if os.path.exists(new_path):
            print(f"Skipping (already exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

def rename_0449_lecture_files(base_dir="json_output/cs0449"):
    for filename in os.listdir(base_dir):
        if not filename.endswith(".json"):
            continue

        # match leading digits at start of filename
        match = re.match(r"(\d+)", filename)
        if not match:
            continue

        lec_num = int(match.group(1))
        new_name = f"lec{lec_num:02d}.json"

        old_path = os.path.join(base_dir, filename)
        new_path = os.path.join(base_dir, new_name)

        if os.path.exists(new_path):
            print(f"Skipping (already exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")
        
def rename_1502_lecture_files(base_dir="json_output/cs1502"):
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue

            name_only = filename[:-5]  # remove .json

            # Match:
            # 2 - something
            # 12 - something
            # 12-5 - something
            match = re.match(r"^(\d+)(?:-(\d+))?\s*-", name_only)
            if not match:
                print(f"Skipping (no match): {filename}")
                continue

            whole = match.group(1)     # always digits
            decimal = match.group(2)   # digits or None

            if decimal:
                lec_label = f"{whole}.{decimal}"
            else:
                lec_label = f"{int(whole):02d}"

            new_name = f"lec{lec_label}.json"

            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_name)

            if os.path.exists(new_path):
                print(f"Skipping (exists): {new_path}")
                continue

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_name}")
            
def flatten_json_directory(base_dir):
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for file in files:
            if not file.lower().endswith(".json"):
                continue

            src_path = os.path.join(root, file)
            dst_path = os.path.join(base_dir, file)

            # Skip if already in root
            if src_path == dst_path:
                continue

            # Handle name collisions
            if os.path.exists(dst_path):
                print(f"Skipping duplicate: {file}")
                continue

            shutil.move(src_path, dst_path)

        # Remove empty directories
        if root != base_dir and not os.listdir(root):
            os.rmdir(root)

def rename_1550_lecture_files(base_dir="json_output/cs1550"):
    pattern = re.compile(r"^Lecture\s*(\d+)", re.IGNORECASE)

    for filename in os.listdir(base_dir):
        if not filename.lower().endswith(".json"):
            continue

        match = pattern.match(filename)
        if not match:
            print(f"Skipping (no match): {filename}")
            continue

        lec_num = int(match.group(1))
        new_name = f"lec{lec_num:02d}.json"

        old_path = os.path.join(base_dir, filename)
        new_path = os.path.join(base_dir, new_name)

        if os.path.exists(new_path):
            print(f"Skipping (already exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")
        
def rename_1622_lecture_files(base_dir="json_output/cs1622"):
    """
    Renames CS 1622 lecture JSON files to:
    lecXX.json or lecX.Y.json
    """

    # Matches:
    # 5 - something.json
    # 06 - something.json
    # 12-5 - something.json
    pattern = re.compile(r"^(\d{1,2})(?:-(\d))?\s*-")

    for filename in os.listdir(base_dir):
        if not filename.lower().endswith(".json"):
            continue

        match = pattern.match(filename)
        if not match:
            print(f"Skipping (no match): {filename}")
            continue

        major = match.group(1)   # e.g. "5", "06", "12"
        minor = match.group(2)   # e.g. "5" in 12-5

        if minor:
            lec_label = f"{int(major)}.{minor}"
        else:
            lec_label = f"{int(major):02d}"

        new_name = f"lec{lec_label}.json"

        old_path = os.path.join(base_dir, filename)
        new_path = os.path.join(base_dir, new_name)

        if os.path.exists(new_path):
            print(f"Skipping (already exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")
        
def truncate_1622_text_filenames(base_dir="jsonl_files_text"):
    """
    Renames files like:
    cs1622_lec05_top_down_parsing.txt
    → cs1622_lec05.txt
    """

    pattern = re.compile(r"^(cs1622_lec\d{2})")

    for filename in os.listdir(base_dir):
        if not filename.lower().endswith(".txt"):
            continue

        match = pattern.match(filename)
        if not match:
            print(f"Skipping (no match): {filename}")
            continue

        base_name = match.group(1)
        new_name = f"{base_name}.txt"

        old_path = os.path.join(base_dir, filename)
        new_path = os.path.join(base_dir, new_name)

        if filename == new_name:
            continue

        if os.path.exists(new_path):
            print(f"Skipping (already exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")



def build_training_jsonl(
    json_output_dir="json_output",
    text_dir="jsonl_files_text",
    output_file="training_dataset.jsonl"
):
    """
    Builds a JSONL training dataset by matching:
    json_output/csXXXX/lecXX.json
    with
    jsonl_files_text/csXXXX_lecXX.txt
    """

    datapoint_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for course in os.listdir(json_output_dir):
            course_path = os.path.join(json_output_dir, course)

            if not os.path.isdir(course_path):
                continue

            course = course.lower()  # safety

            for filename in os.listdir(course_path):
                if not filename.lower().endswith(".json"):
                    continue

                lec_name = filename.replace(".json", "")  # lec01, lec12.5
                key = f"{course}_{lec_name}"

                text_file = f"{key}.txt"
                text_path = os.path.join(text_dir, text_file)

                if not os.path.exists(text_path):
                    continue  # no labels → skip

                # Load lecture JSON
                with open(os.path.join(course_path, filename), "r", encoding="utf-8") as jf:
                    lecture_data = json.load(jf)

                # Load concept text
                with open(text_path, "r", encoding="utf-8") as tf:
                    concepts = tf.read().strip()

                datapoint = {
                    "id": key,
                    "input": lecture_data,
                    "concepts": concepts
                }

                out_f.write(json.dumps(datapoint) + "\n")
                datapoint_count += 1

    print(f"✅ Created {datapoint_count} training datapoints")


if __name__ == "__main__":
    #0007 -> good
    #0441 -> good
    #0447 -> good
    #0449 -> good
    #1502 -> good
    #1550 -> good
    #1622 -> good
    change_title_name()
    rename_lecture_files_007()
    rename_0447_lecture_files()
    rename_0449_lecture_files()
    rename_1502_lecture_files()
    flatten_json_directory("json_output/cs1502")
    rename_1550_lecture_files()
    rename_1622_lecture_files()
    truncate_1622_text_filenames()
    build_training_jsonl()