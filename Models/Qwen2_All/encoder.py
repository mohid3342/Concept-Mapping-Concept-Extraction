import re

CS_KEYWORDS = [
    "algorithm", "data structure", "tree", "graph", "hash",
    "recursion", "dynamic programming", "sorting", "search",
    "complexity", "queue", "stack", "heap", "binary",
    "compiler", "database", "sql", "memory", "concurrency",
    "machine learning", "ai", "network"
]

ADMIN_PATTERNS = [
    r"@\w+",
    r"office hours",
    r"instructor",
    r"ta",
    r"grading",
    r"midterm",
    r"final",
    r"http",
    r"www"
]

def compute_salience(text: str) -> float:
    t = text.lower()

    if any(re.search(p, t) for p in ADMIN_PATTERNS):
        return 0.05

    score = 0.0
    for kw in CS_KEYWORDS:
        if kw in t:
            score += 0.35

    if text.strip().startswith("-") or text.strip().startswith("•"):
        score += 0.1

    return min(score, 1.0)


def encode_lecture(lecture_text: str) -> str:
    slides = re.split(r"Slide\s*\d+:", lecture_text)

    out = []

    for i, slide in enumerate(slides):
        if not slide.strip():
            continue

        out.append(f"[SLIDE {i}]")

        for line in slide.split("\n"):
            line = line.strip()
            if not line:
                continue

            score = compute_salience(line)

            if score >= 0.75:
                tag = "HIGH"
            elif score >= 0.4:
                tag = "MED"
            else:
                tag = "LOW"

            out.append(f"[{tag} | score={score:.2f}] {line}")

        out.append("[END_SLIDE]\n")

    return "\n".join(out)