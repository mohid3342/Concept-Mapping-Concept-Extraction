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
                {"type": "input_text", "text": "Describe what is in this image in 30 words"},
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