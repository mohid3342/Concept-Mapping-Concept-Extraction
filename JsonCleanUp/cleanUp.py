import json
import re

input_file = "all.jsonl"
output_file = "cleaned.txt"

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        obj = json.loads(line)

        text = obj["text"]
        label = obj["label"]

        # Clean text -> single line
        cleaned_text = re.sub(r'[\n\r\f\t]+', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Write output
        outfile.write(cleaned_text + "\n")
        outfile.write(json.dumps(label) + "\n\n")

print("Cleaning complete!")
