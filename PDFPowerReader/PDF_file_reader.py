import fitz  # PyMuPDF
import json
import base64
from openai import OpenAI


class PDFToJSONConverter:
    def __init__(self):
        self.client = OpenAI(
            #api_key=
        )

    # --------------------------------------------------
    # Feature 1 — Font size normalization
    # --------------------------------------------------
    def normalize_font_size(self, size):
        if size is None:
            return None
        return int(round(size))

    # --------------------------------------------------
    # Feature 2 — Run normalization
    # --------------------------------------------------
    def normalize_run(self, run):
        font_name_lower = (run.get("font_name") or "").lower()

        # Infer styles from font name
        is_bold = "bold" in font_name_lower
        is_italic = "italic" in font_name_lower

        # Normalize font family name
        base_font = run.get("font_name", "")
        for suffix in ["-BoldItalic", "-Bold", "-Italic"]:
            base_font = base_font.replace(suffix, "")

        return {
            **run,
            "bold": is_bold,
            "italic": is_italic,
            "font_name": base_font,
            "font_size": self.normalize_font_size(run.get("font_size"))
        }

    # --------------------------------------------------
    # Image helpers
    # --------------------------------------------------
    def image_to_base64(self, pix):
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")

    def describe_image_with_ai(self, image_base64):
        response = self.client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Describe this image for a college lecture slide in 25 words"
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_base64}",
                            "detail": "low"
                        }
                    ]
                }
            ]
        )
        return response.output_text

    def generate_alt_text_with_ai(self, image_base64):
        response = self.client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Write concise accessibility alt text for this image (max 15 words)"
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_base64}",
                            "detail": "low"
                        }
                    ]
                }
            ]
        )
        return response.output_text

    # --------------------------------------------------
    # Underline detection (PDF heuristic)
    # --------------------------------------------------
    def is_underlined(self, span_bbox, drawings, tolerance=2):
        x0, y0, x1, y1 = span_bbox

        for d in drawings:
            for item in d["items"]:
                if item[0] != "l":
                    continue

                _, p1, p2 = item

                # Must be horizontal
                if abs(p1.y - p2.y) > 1:
                    continue

                # Must be just below text
                if abs(p1.y - y1) <= tolerance:
                    if p1.x <= x1 and p2.x >= x0:
                        return True

        return False

    # --------------------------------------------------
    # Core extraction
    # --------------------------------------------------
    def extract_pdf_data(self, pdf_path):
        doc = fitz.open(pdf_path)
        slides = []

        for page_index, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            drawings = page.get_drawings()

            text_blocks = []
            title = None
            max_font = 0

            # ---------------- TEXT ----------------
            for block in page_dict["blocks"]:
                if block["type"] != 0:
                    continue

                block_group = []

                for line in block["lines"]:
                    runs = []
                    full_text = ""

                    for span in line["spans"]:
                        text = span["text"]
                        if not text.strip():
                            continue

                        flags = span["flags"]
                        bbox = span["bbox"]

                        underline = self.is_underlined(bbox, drawings)

                        run = {
                            "text": text,
                            "bold": bool(flags & 2),
                            "italic": bool(flags & 1),
                            "underline": underline,
                            "font_name": span.get("font"),
                            "font_size": round(span["size"], 2)
                        }

                        # Normalize run (Feature integration)
                        run = self.normalize_run(run)

                        runs.append(run)
                        full_text += text

                        # Infer title from largest font
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

            # ---------------- IMAGES ----------------
            images = []
            image_list = page.get_images(full=True)

            for img in image_list:
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_b64 = self.image_to_base64(pix)

                    alt_text = self.generate_alt_text_with_ai(img_b64)
                    ai_desc = self.describe_image_with_ai(img_b64)

                    images.append({
                        "alt_text": alt_text,
                        "ai_description": ai_desc
                    })

                    pix = None

                except Exception as e:
                    images.append({
                        "alt_text": "",
                        "ai_description": None,
                        "error": str(e)
                    })

            slides.append({
                "slide_number": page_index,
                "title": title,
                "text_blocks": text_blocks,
                "images": images,
                "notes": None
            })

        return slides

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------
    def convert_pdf_to_json(self, pdf_path, output_path):
        pdf_structure = self.extract_pdf_data(pdf_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pdf_structure, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved: {output_path}")


# --------------------------------------------------
# RUNNER
# --------------------------------------------------
if __name__ == "__main__":
    converter = PDFToJSONConverter()

    converter.convert_pdf_to_json(
        pdf_path="test.pdf",
        output_path="output8.json"
    )
    