from pypdf import PdfReader
import json
import base64
from openai import OpenAI

# ======================================================
# OpenAI Client (hard-coded key)
# ======================================================
client = OpenAI(
    #api_key=
)

# ======================================================
# Image Helpers
# ======================================================
def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def describe_image_with_ai(image_base64):
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image for a college lecture slide in under 25 words."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                ]
            }
        ]
    )

    return response.output_text.strip() if response.output_text else ""

# ======================================================
# AI Slide Structuring (JSON-LOCKED)
# ======================================================
def structure_slide_text(raw_text):
    """
    Converts raw PDF text into structured slide JSON.
    HARD guarantees valid JSON.
    """

    prompt = f"""
Return ONLY valid JSON.
No markdown. No explanations. No backticks.

Schema (must match exactly):

{{
  "title": string | null,
  "text_blocks": [
    [
      {{
        "text": string,
        "runs": [
          {{
            "text": string,
            "bold": boolean | null,
            "italic": boolean | null,
            "underline": boolean | null,
            "font_name": null,
            "font_size": null
          }}
        ]
      }}
    ]
  ]
}}

Rules:
- Infer title from short prominent lines
- Use bold=true for headings or emphasized concepts
- Preserve bullet order
- Do NOT invent content
- If nothing applies, return empty arrays or nulls

RAW TEXT:
{raw_text}
"""

    response = client.responses.create(
        model="gpt-4o",
        response_format={"type": "json_object"},  # 🔒 GUARANTEE JSON
        input=prompt
    )

    # The SDK already parsed it for us
    return response.output_parsed

# ======================================================
# PDF Extraction
# ======================================================
def extract_pdf_data(file_path):
    reader = PdfReader(file_path)
    pages = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_info = {
            "slide_number": page_index,
            "title": None,
            "text_blocks": [],
            "images": []
        }

        # ------------------ TEXT ------------------
        raw_text = page.extract_text()
        if raw_text and raw_text.strip():
            try:
                structured = structure_slide_text(raw_text)
                page_info["title"] = structured.get("title")
                page_info["text_blocks"] = structured.get("text_blocks", [])
            except Exception as e:
                page_info["text_blocks"] = []
                page_info["title"] = None
                page_info["error"] = f"Text parse failed: {str(e)}"

        # ------------------ IMAGES ------------------
        resources = page.get("/Resources", {})
        if "/XObject" in resources:
            xobjects = resources["/XObject"].get_object()
            for obj in xobjects:
                xobj = xobjects[obj]
                if xobj.get("/Subtype") == "/Image":
                    try:
                        image_bytes = xobj.get_data()
                        image_base64 = image_to_base64(image_bytes)
                        ai_description = describe_image_with_ai(image_base64)

                        page_info["images"].append({
                            "alt_text": "",
                            "ai_description": ai_description
                        })
                    except Exception as e:
                        page_info["images"].append({
                            "alt_text": "",
                            "ai_description": None,
                            "error": str(e)
                        })

        pages.append(page_info)

    # ------------------ METADATA ------------------
    meta = reader.metadata or {}
    metadata = {
        "title": meta.get("/Title"),
        "author": meta.get("/Author"),
        "subject": meta.get("/Subject"),
        "creator": meta.get("/Creator"),
        "producer": meta.get("/Producer"),
        "creation_date": meta.get("/CreationDate"),
        "modification_date": meta.get("/ModDate")
    }

    return {
        "metadata": metadata,
        "pages": pages
    }

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    input_pdf = "Lecture 0.pdf"
    output_json = "pdf_test_data.json"

    pdf_structure = extract_pdf_data(input_pdf)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pdf_structure, f, indent=2, ensure_ascii=False)

    print(f"Structured data saved to {output_json}")
