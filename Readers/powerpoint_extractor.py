import os
from file_reader import PowerPointToJSONConverter


DATA_SOURCE_FOLDER = "data_source"
OUTPUT_FOLDER = "json_output"

converter = PowerPointToJSONConverter()

for root, dirs, files in os.walk(DATA_SOURCE_FOLDER):
    for file in files:
        if file.lower().endswith(".pptx"):
            pptx_path = os.path.join(root, file)

            # Preserve folder structure
            relative_path = os.path.relpath(root, DATA_SOURCE_FOLDER)
            output_dir = os.path.join(OUTPUT_FOLDER, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            json_filename = os.path.splitext(file)[0] + ".json"
            output_path = os.path.join(output_dir, json_filename)

            try:
                converter.convert_pptx_to_json(pptx_path, output_path)
            except Exception as e:
                print(f"❌ Failed: {pptx_path} — {e}")
