python convert_hf_to_gguf.py ../models/tinyllama_1_1b/ --outfile ../gguf_models/tinyllama.gguf --outtype f16
./build/bin/llama-eval-callback -m ../gguf_models/qwen.gguf -p "hello" -n 1 > ../gguf_models/qwen_out.txt
python get_ops.py gguf_models/tiny_out.txt
