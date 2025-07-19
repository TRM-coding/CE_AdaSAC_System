for x in $(seq 0 0.1 0.9); do
  python GPTJ_generate_SVD_layer.py "$x"
done
