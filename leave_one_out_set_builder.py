import os
import json

JSON_DIR = "json_output"
LABELED_DIR = "labeled_plus_fd"
OUTPUT_DIR = "leave_one_out_testing"

instruction_text = (
    "You extract core computer science concepts from lecture JSON data. "
    "Return only a newline-separated list of unique concepts. "
    "Do not include explanations or extra text. If there are not core important concepts, return 'None'."
)

def extract_concepts_per_slide(jsonl_path, lecture_json):
    """Returns a dict mapping slide_number -> list of concepts"""
    slide_concepts = {}
    
    # First, extract all concepts from the labeled file
    all_concepts = []
    
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
                    all_concepts.append(concept_text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_concepts = []
    for c in all_concepts:
        if c not in seen:
            seen.add(c)
            unique_concepts.append(c)
    
    # Now match concepts to slides by checking if the concept text appears in the slide
    for slide in lecture_json:
        slide_num = slide.get("slide_number")
        if slide_num is None:
            continue
        
        # Build the slide text from all text blocks
        slide_text = ""
        for text_block_list in slide.get("text_blocks", []):
            for text_block in text_block_list:
                slide_text += text_block.get("text", "")
        
        # Check which concepts appear in this slide
        slide_concepts[slide_num] = []
        for concept in unique_concepts:
            if concept in slide_text:
                slide_concepts[slide_num].append(concept)
    
    return slide_concepts


def process_course(course):
    """Process a single course and return list of training examples"""
    examples = []
    course_path = os.path.join(JSON_DIR, course)
    
    for lecture_file in os.listdir(course_path):
        if not lecture_file.endswith(".json"):
            continue

        lec_name = lecture_file.replace(".json", "")
        json_path = os.path.join(course_path, lecture_file)

        labeled_path = os.path.join(
            LABELED_DIR, course, f"{lec_name}.jsonl"
        )

        if not os.path.exists(labeled_path):
            print(f"  Skipping {course}/{lec_name} (no labeled file)")
            continue

        print(f"  Processing {course}/{lec_name}")

        # Load lecture JSON
        with open(json_path, "r", encoding="utf-8") as jf:
            lecture_json = json.load(jf)

        # Extract labeled concepts per slide
        slide_concepts = extract_concepts_per_slide(labeled_path, lecture_json)

        # Process slides in pairs
        for i in range(0, len(lecture_json), 2):
            # Get current slide and next slide (if exists)
            slides_batch = lecture_json[i:i+2]
            
            # Collect all concepts from both slides
            all_concepts = []
            for slide in slides_batch:
                slide_num = slide.get("slide_number")
                if slide_num is not None:
                    concepts = slide_concepts.get(slide_num, [])
                    all_concepts.extend(concepts)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_concepts = []
            for c in all_concepts:
                if c not in seen:
                    seen.add(c)
                    unique_concepts.append(c)

            # Determine output
            if unique_concepts:
                output_content = "\n".join(unique_concepts)
            else:
                output_content = "None"

            training_example = {
                "instruction": instruction_text,
                "input": json.dumps(slides_batch, ensure_ascii=False),
                "output": output_content
            }

            examples.append(training_example)

        num_batches = (len(lecture_json) + 1) // 2
        print(f"    Added {num_batches} batches from {len(lecture_json)} slides")
    
    return examples


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all courses
all_courses = [c for c in os.listdir(JSON_DIR) if os.path.isdir(os.path.join(JSON_DIR, c))]

print(f"Found {len(all_courses)} courses: {all_courses}\n")

# For each course, create a fold where that course is the test set
for test_course in all_courses:
    print(f"\n{'='*60}")
    print(f"FOLD: Holding out {test_course}")
    print(f"{'='*60}")
    
    train_courses = [c for c in all_courses if c != test_course]
    
    # Process test course
    print(f"\nProcessing TEST course: {test_course}")
    test_examples = process_course(test_course)
    
    # Process training courses
    train_examples = []
    print(f"\nProcessing TRAIN courses: {train_courses}")
    for train_course in train_courses:
        print(f"\nProcessing TRAIN course: {train_course}")
        train_examples.extend(process_course(train_course))
    
    # Write training file
    train_file = os.path.join(OUTPUT_DIR, f"train_without_{test_course}.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # Write test file
    test_file = os.path.join(OUTPUT_DIR, f"test_{test_course}.jsonl")
    with open(test_file, "w", encoding="utf-8") as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # ============= ADD NEW CODE HERE =============
    # Extract unique concepts from test examples
    unique_concepts = set()
    for example in test_examples:
        output = example.get("output", "")
        if output.strip() != "None":
            concepts = output.split("\n")
            for concept in concepts:
                concept = concept.strip()
                if concept:
                    unique_concepts.add(concept)
    
    # Write concepts to txt file
    concepts_file = os.path.join(OUTPUT_DIR, f"concepts_{test_course}.txt")
    with open(concepts_file, "w", encoding="utf-8") as f:
        for concept in sorted(unique_concepts):
            f.write(concept + "\n")
    # ============= END NEW CODE =============
    
    print(f"\nFold complete:")
    print(f"  Training examples: {len(train_examples)} (from {len(train_courses)} courses)")
    print(f"  Test examples: {len(test_examples)} (from {test_course})")
    print(f"  Unique concepts: {len(unique_concepts)}")
    print(f"  Saved to: {train_file}, {test_file}, and {concepts_file}")

print(f"\n{'='*60}")
print(f"Leave-one-out cross-validation complete!")
print(f"Created {len(all_courses)} folds in '{OUTPUT_DIR}' directory")
print(f"{'='*60}")