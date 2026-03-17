from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

print(pipe("Explain what a neuron is in simple terms:", max_new_tokens=50))
