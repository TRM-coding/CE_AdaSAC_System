from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "/home/tianruiming/sdpcos_2025/code/gpt2/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e",
    local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/home/tianruiming/sdpcos_2025/code/gpt2/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e",
    local_files_only=True,
)

prompt="hello, how are you doing today? I am"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
for i, output in enumerate(outputs):
    print(tokenizer.decode(output, skip_special_tokens=True),end='')