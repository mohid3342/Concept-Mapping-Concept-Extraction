from pypdf import PdfReader
import json
import base64
from openai import OpenAI

client = OpenAI(
    #api_key=
)

def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def describe_image_with_ai(image_base64):
    response = client.responses.create(
        model="gpt-4o",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Describe this image for a college lecture slide in 25 words"
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "low"
                }
            ]
        }]
    )
    return response.output_text

def extract_pdf_data(file_path):
    reader = PdfReader(file_path)
    pdf_data = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_info = {
            "slide_number": page_index,   # kept name for compatibility
            "title": None,
            "text_blocks": [],
            "images": []
        }

        # ---- TEXT EXTRACTION ----
        text = page.extract_text()
        if text:
            text_block = []
            for line in text.splitlines():
                if not line.strip():
                    continue

                para_content = {
                    "text": line,
                    "runs": [{
                        "text": line,
                        "bold": None,
                        "italic": None,
                        "underline": None,
                        "font_name": None,
                        "font_size": None
                    }]
                }
                text_block.append(para_content)

            page_info["text_blocks"].append(text_block)

        # ---- IMAGE EXTRACTION ----
        if "/XObject" in page["/Resources"]:
            xobjects = page["/Resources"]["/XObject"].get_object()
            for obj in xobjects:
                xobj = xobjects[obj]
                if xobj["/Subtype"] == "/Image":
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

        pdf_data.append(page_info)

        # ---- Meta Data Extraction ----
        meta = reader.metadata
        meta_content = {
            "title": meta.get("/Title") if meta else None,
            "author": meta.get("/Author") if meta else None,
            "subject": meta.get("/Subject") if meta else None,
            "creator": meta.get("/Creator") if meta else None,
            "producer": meta.get("/Producer") if meta else None,
            "creation_date": meta.get("/CreationDate") if meta else None,
            "modification_date": meta.get("/ModDate") if meta else None
        }
    
    return {
        "metadata": meta_content,
        "pages": pdf_data
    }


    # return pdf_data



# -----------------------------
# Example usage
file_path = "Lecture 0.pdf"
pdf_structure = extract_pdf_data(file_path)

output_file = "pdf_test_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(pdf_structure, f, indent=2, ensure_ascii=False)

print(f"Data saved to {output_file}")