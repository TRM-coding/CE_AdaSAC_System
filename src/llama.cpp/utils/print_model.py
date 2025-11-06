from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./models/qwen2_5_1_5b"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

print(model)
