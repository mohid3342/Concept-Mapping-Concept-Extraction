import fitz  # PyMuPDF
import json
import base64
from openai import OpenAI
import os


class PDFToJSONConverter:
    def __init__(self):
        self.client = OpenAI(
            api_key="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"
        )

    # ----------------------------
    # Image helpers
    # ----------------------------
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

    # ----------------------------
    # Core extraction
    # ----------------------------
    def extract_pdf_data(self, pdf_path):
        doc = fitz.open(pdf_path)
        slides = []

        for page_index, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")

            text_blocks = []
            title = None
            max_font = 0

            # -------- TEXT --------
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

                        run = {
                            "text": text,
                            "bold": bool(flags & 2),
                            "italic": bool(flags & 1),
                            "underline": bool(flags & 4) if flags & 4 else None,
                            "font_name": span.get("font"),
                            "font_size": round(span["size"], 2)
                        }

                        runs.append(run)
                        full_text += text

                        # Title inference (largest font on page)
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

            # -------- IMAGES --------
            images = []
            image_list = page.get_images(full=True)

            for img in image_list:
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_b64 = self.image_to_base64(pix)
                    ai_desc = self.describe_image_with_ai(img_b64)

                    images.append({
                        "alt_text": "",
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

    # ----------------------------
    # Save JSON
    # ----------------------------
    def convert_pdf_to_json(self, pdf_path, output_path):
        pdf_structure = self.extract_pdf_data(pdf_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pdf_structure, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    converter = PDFToJSONConverter()

    converter.convert_pdf_to_json(
        pdf_path="test.pdf",
        output_path="output2.json"
    )
