from openai import OpenAI
import base64


client = OpenAI(api_key="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")
with open("test_bio_image.png", "rb") as f:
    image_bytes = f.read()

image_base64 = base64.b64encode(image_bytes).decode("utf-8")

response = client.responses.create(
    model="gpt-4o",  # vision-capable model
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe what is in this image"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "high"
                }
            ]
        }
    ]
)

print(response.output_text)