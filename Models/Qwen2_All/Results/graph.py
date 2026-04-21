import json
import matplotlib.pyplot as plt
import os

# Put your JSON file paths here
files = [
    "evaluation_results_cs0007_testCode2.json",
    "evaluation_results_cs0441_testCode2.json",
    "evaluation_results_cs0449_testCode2.json",
    "evaluation_results_cs1502_testCode2.json",
    "evaluation_results_cs1550_testCode2.json"
]

# Store results
names = []
precision = []
recall = []
f1 = []

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        metrics = data["metrics"]

        # Use filename as label
        names.append(os.path.basename(file))

        precision.append(metrics["precision"])
        recall.append(metrics["recall"])
        f1.append(metrics["f1"])

# X positions
x = range(len(names))

# Plot
plt.figure(figsize=(10, 6))

plt.plot(x, precision, marker='o', label='Precision')
plt.plot(x, recall, marker='o', label='Recall')
plt.plot(x, f1, marker='o', label='F1 Score')

plt.xticks(x, names, rotation=30)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()

plt.show()