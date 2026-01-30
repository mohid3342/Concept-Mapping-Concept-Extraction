import fitz  # PyMuPDF
import json

doc = fitz.open("test.pdf")
slides = []

for page_index, page in enumerate(doc, start=1):
    page_dict = page.get_text("dict")

    text_blocks = []
    title = None
    max_font = 0

    for block in page_dict["blocks"]:
        if block["type"] != 0:
            continue  # not text

        block_group = []

        for line in block["lines"]:
            runs = []
            full_text = ""

            for span in line["spans"]:
                text = span["text"]
                if not text.strip():
                    continue

                flags = span["flags"]

                run = {
                    "text": text,
                    "bold": bool(flags & 2),
                    "italic": bool(flags & 1),
                    "underline": bool(flags & 4) if flags & 4 else None,
                    "font_name": span["font"],
                    "font_size": round(span["size"], 2)
                }

                runs.append(run)
                full_text += text

                # infer title
                if span["size"] > max_font:
                    max_font = span["size"]
                    title = text.strip()

            if runs:
                block_group.append({
                    "text": full_text.strip(),
                    "runs": runs
                })

        if block_group:
            text_blocks.append(block_group)

    slides.append({
        "slide_number": page_index,
        "title": title,
        "text_blocks": text_blocks,
        "images": [],
        "notes": None
    })

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(slides, f, indent=2, ensure_ascii=False)
