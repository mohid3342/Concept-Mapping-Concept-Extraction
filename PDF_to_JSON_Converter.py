import os
from PDF_file_reader import PDFToJSONConverter

DATA_SOURCE_FOLDER = "data_source"
OUTPUT_FOLDER = "json_output"
PROGRESS_FILE = "progress.txt"  # stores last processed PDF

converter = PDFToJSONConverter(openai_api_key="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")

# Read last processed file (if any)
last_processed = None
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        last_processed = f.read().strip()

skip_files = True if last_processed else False


TARGET_SUBDIRS = [
    os.path.join(DATA_SOURCE_FOLDER, "CS-1550 Lecture Slides"),
]

for start_dir in TARGET_SUBDIRS:
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue
            print( f"Processing file: {file} in {root}" )
            pdf_path = os.path.join(root, file)

            # Skip files until we reach last processed
            if skip_files:
                if pdf_path == last_processed:
                    skip_files = False
                continue

            # Preserve folder structure
            relative_path = os.path.relpath(root, DATA_SOURCE_FOLDER)
            output_dir = os.path.join(OUTPUT_FOLDER, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            json_filename = os.path.splitext(file)[0] + ".json"
            output_path = os.path.join(output_dir, json_filename)
            print("We are going to convert:", pdf_path)
            try:
                # Skip if already converted
                if os.path.exists(output_path):
                    print(f"⏭️ Already exists, skipping: {output_path}")
                    continue

                converter.convert_pdf_to_json(pdf_path, output_path)

                # Save progress
                with open(PROGRESS_FILE, "w") as f:
                    f.write(pdf_path)
                    print(f"✅ Converted: {pdf_path} to {output_path}")

            # Ask user to continue
            

            except Exception as e:
                print(f"❌ Failed: {pdf_path} — {e}")
