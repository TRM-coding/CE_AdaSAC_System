from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

cache_dir = "/SSD/trm/llama2"

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir=cache_dir, device_map="auto")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",cache_dir="/SSD/trm/llama2")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",cache_dir="/SSD/trm/llama2")

# print(model)
print(transformers.__file__)