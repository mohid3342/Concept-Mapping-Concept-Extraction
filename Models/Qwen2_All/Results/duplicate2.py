import json
import os
from collections import Counter

# Your JSON files
files = [
    "evaluation_results_cs0007_testCode2.json",
    "evaluation_results_cs0441_testCode2.json",
    "evaluation_results_cs0449_testCode2.json",
    "evaluation_results_cs1502_testCode2.json",
    "evaluation_results_cs1550_testCode2.json"
]

# Store all predicted concepts across the files
all_predicted_concepts = []

# Loop over each file and collect its predicted concepts
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        predicted_concepts = data["predicted_concepts"]
        all_predicted_concepts.extend(predicted_concepts)

# Count occurrences of each concept
concept_counts = Counter(all_predicted_concepts)

# Find duplicates (concepts that appear more than 4 times)
duplicates = {concept: count for concept, count in concept_counts.items() if count >= 4}

# Print the duplicates and their counts
if duplicates:
    print("Duplicated Concepts (4 or more occurrences) and Their Counts:")
    for concept, count in duplicates.items():
        print(f"{concept}: {count}")
else:
    print("No duplicates found with 4 or more occurrences.")