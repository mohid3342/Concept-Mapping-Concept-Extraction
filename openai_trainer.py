from openai import OpenAI

client = OpenAI(api_key="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")

# 1. Upload dataset
file = client.files.create(
    file=open("concept_extraction_training_openai_per_slide.jsonl", "rb"),
    purpose="fine-tune"
)

print("File ID:", file.id)

# 2. Create fine-tuning job
fine_tune = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-2024-08-06",   # ✅ FIXED MODEL
    suffix="concept-mapping-v1"
)

print("Fine-tune job:", fine_tune.id)

# 3. Check status
job = client.fine_tuning.jobs.retrieve(fine_tune.id)
print("Status:", job.status)