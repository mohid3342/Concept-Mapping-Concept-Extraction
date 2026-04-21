import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Your JSON files
files = [
    "evaluation_results_cs0007_testCode2.json",
    "evaluation_results_cs0441_testCode2.json",
    "evaluation_results_cs0449_testCode2.json",
    "evaluation_results_cs1502_testCode2.json",
    "evaluation_results_cs1550_testCode2.json"
]

names = ["CS0007","CS0441","CS0449","CS1502","CS1550"]
true_positives = []
false_positives = []
false_negatives = []
ground_truth_count = []

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        metrics = data["metrics"]

        true_positives.append(metrics["true_positives"])
        false_positives.append(metrics["false_positives"])
        false_negatives.append(metrics["false_negatives"])
        ground_truth_count.append(metrics["ground_truth_count"])

# X positions
x = np.arange(len(names))
width = 0.2

# Plot
plt.figure(figsize=(10, 8))

plt.bar(x - 1.5*width, true_positives, width, label='True Positives')
plt.bar(x - 0.5*width, false_positives, width, label='False Positives')
plt.bar(x + 0.5*width, false_negatives, width, label='False Negatives')
plt.bar(x + 1.5*width, ground_truth_count, width, label='Ground Truth Count')

plt.xticks(x, names, rotation=30)
plt.ylabel("Count")
plt.title("QWEN 2 Results")
plt.legend()

plt.tight_layout()
plt.show()