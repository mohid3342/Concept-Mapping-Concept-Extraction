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
precision = []
recall = []
f1 = []

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        metrics = data["metrics"]

        
        precision.append(metrics["precision"])
        recall.append(metrics["recall"])
        f1.append(metrics["f1"])

# X positions
x = np.arange(len(names))
width = 0.25

# Plot
plt.figure(figsize=(10, 6))

plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1 Cars')

plt.xticks(x, names, rotation=30)
plt.ylabel("Score")
plt.title("QWEN 2 Results")
plt.legend()

plt.tight_layout()
plt.show()