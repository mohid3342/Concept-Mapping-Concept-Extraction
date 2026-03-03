import os
import json

JSON_DIR = "json_output"
LABELED_DIR = "labeled_plus_fd"
OUTPUT_FILE = "concept_extraction_training_openai_per_slide.jsonl"

system_prompt = (
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

            print(f"\nProcessing {course}/{lec_name}")

            # Load lecture JSON
            with open(json_path, "r", encoding="utf-8") as jf:
                lecture_json = json.load(jf)

            # Extract labeled concepts per slide
            slide_concepts = extract_concepts_per_slide(labeled_path, lecture_json)
            
            print(f"  Found concepts in {len(slide_concepts)} slides")

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

                # Determine assistant response
                if unique_concepts:
                    assistant_content = "\n".join(unique_concepts)
                else:
                    assistant_content = "None"

                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": json.dumps(slides_batch, ensure_ascii=False)
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content
                        }
                    ]
                }

                out.write(json.dumps(training_example, ensure_ascii=False) + "\n")

            num_batches = (len(lecture_json) + 1) // 2
            print(f"Added: {course}/{lec_name} ({num_batches} batches from {len(lecture_json)} slides)")

print("\nOpenAI training file complete (per-slide).")