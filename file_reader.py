from pptx import Presentation
import json
from pptx.enum.shapes import MSO_SHAPE_TYPE


def get_alt_text(shape):
    try:
        cNvPr = shape._element.xpath(".//*[local-name() = 'cNvPr']")
        if cNvPr:
            title = cNvPr[0].get("title")
            descr = cNvPr[0].get("descr")
            if descr and descr.strip():
                return descr.strip()
            if title and title.strip():
                return title.strip()
    except Exception:
        pass
    return ""

def extract_pptx_data(file_path):
    prs = Presentation(file_path)
    ppt_data = []

    for slide_index, slide in enumerate(prs.slides, start=1):
        slide_info = {
            "slide_number": slide_index,
            "title": slide.shapes.title.text if slide.shapes.title else None,
            "text_blocks": [],
            "images": []
        }

        for shape in slide.shapes:
            # Text shapes
            if shape.has_text_frame:
                text_block = []
                for paragraph in shape.text_frame.paragraphs:
                    para_content = {
                        "text": paragraph.text,
                        "runs": []
                    }
                    for run in paragraph.runs:
                        run_info = {
                            "text": run.text,
                            "bold": run.font.bold,
                            "italic": run.font.italic,
                            "underline": run.font.underline,
                            "font_name": run.font.name,
                            "font_size": run.font.size.pt if run.font.size else None
                        }
                        para_content["runs"].append(run_info)
                    text_block.append(para_content)
                slide_info["text_blocks"].append(text_block)

            # Images with alt text
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:  # Picture
                try:
                    alt_text = get_alt_text(shape)
                    slide_info["images"].append({
                        "alt_text": alt_text
                    })
                except Exception:
                    slide_info["images"].append({
                        "alt_text": "No alt text"
                    })

        ppt_data.append(slide_info)

    return ppt_data

# -----------------------------
# Example usage
file_path = "TPL_Chapter_1.pptx"
ppt_structure = extract_pptx_data(file_path)

# Save to a JSON file
output_file = "ppt_test_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(ppt_structure, f, indent=2, ensure_ascii=False)

print(f"Data saved to {output_file}")

