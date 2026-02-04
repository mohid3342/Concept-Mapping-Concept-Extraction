import os
import fitz  # PyMuPDF
import json
import base64
from openai import OpenAI


class PDFToJSONConverter:
    def __init__(self, openai_api_key=None):
        self.client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))

    def image_to_base64(self, pix):
        """Convert PyMuPDF Pixmap to base64 string"""
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")

    def describe_image_with_ai(self, image_base64):
        response = self.client.responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image for a college lecture slide in 25 words"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}", "detail": "low"}
                ]
            }]
        )
        return response.output_text

    def extract_pdf_data(self, pdf_path):
        doc = fitz.open(pdf_path)
        slides = []

        for page_index, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            drawings = page.get_drawings()

            # Extract text
            text_blocks = []
            title = None
            max_font = 0

            for block in page_dict["blocks"]:
                if block["type"] != 0:  # skip non-text blocks
                    continue
                block_group = []

                for line in block["lines"]:
                    runs = []
                    full_text = ""
                    for span in line["spans"]:
                        text = span["text"]
                        if not text.strip():
                            continue
                        font_size = round(span["size"], 2)
                        run = {
                            "text": text,
                            "bold": bool(span["flags"] & 2),
                            "italic": bool(span["flags"] & 1),
                            "underline": False,  # could implement detection like your friend
                            "font_name": span.get("font"),
                            "font_size": font_size
                        }
                        runs.append(run)
                        full_text += text

                        # pick largest font as page title
                        if font_size > max_font:
                            max_font = font_size
                            title = text.strip()

                    if runs:
                        block_group.append({"text": full_text.strip(), "runs": runs})

                if block_group:
                    text_blocks.append(block_group)

            # Extract images
            images = []
            for img in page.get_images(full=True):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # handle CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_b64 = self.image_to_base64(pix)
                    ai_desc = self.describe_image_with_ai(img_b64)
                    images.append({"alt_text": "", "ai_description": ai_desc})
                    pix = None
                except Exception as e:
                    images.append({"alt_text": "", "ai_description": None, "error": str(e)})

            slides.append({
                "slide_number": page_index,
                "title": title,
                "text_blocks": text_blocks,
                "images": images,
                "notes": None
            })

        return slides

    def convert_pdf_to_json(self, pdf_path, output_path):
        pdf_structure = self.extract_pdf_data(pdf_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pdf_structure, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {output_path}")

    