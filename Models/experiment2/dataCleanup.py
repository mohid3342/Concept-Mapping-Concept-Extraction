import json
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove escaped unicode junk
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)

    # Remove JSON-like fragments (key:value)
    text = re.sub(r'\w+\s*:\s*\w+', '', text)

    # Remove brackets, quotes, braces
    text = re.sub(r'[\[\]\{\}\"\'`]', '', text)

    # Remove leftover punctuation (keep words + spaces)
    text = re.sub(r'[^a-z0-9\s\-\(\)]', '', text)

    # Remove extra whitespace
    text = text.strip()

    return text


def is_valid_concept(text):
    # Filter out garbage
    if not text:
        return False
    if len(text) < 2:
        return False
    if text in {"null", "true", "false"}:
        return False
    if "runs" in text or "text" in text:
        return False
    return True


def clean_concepts(concepts):
    cleaned = set()

    for item in concepts:
        text = clean_text(item)

        # Sometimes multiple words get smashed → split heuristically
        parts = re.split(r'\s{2,}', text)

        for part in parts:
            part = part.strip()
            if is_valid_concept(part):
                cleaned.add(part)

    return sorted(cleaned)


# ---- LOAD YOUR JSON ----
with open("evaluation_results_cs0007.json", "r") as f:
    data = json.load(f)

# ---- CLEAN ----
cleaned_predictions = clean_concepts(data["predicted_concepts"])

# ---- SAVE OUTPUT ----
output = {
    "cleaned_predicted_concepts": cleaned_predictions
}

with open("cleaned_output.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Cleaned {len(cleaned_predictions)} concepts.")
