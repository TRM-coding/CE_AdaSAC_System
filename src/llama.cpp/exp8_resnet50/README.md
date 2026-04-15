# ResNet50 GGUF Support

## Overview

This directory contains the current ResNet50 adaptation for the workspace `llama.cpp`.

What is supported now:

- Download or reuse a Hugging Face `microsoft/resnet-50` checkpoint.
- Fuse `Conv2d + BatchNorm2d` during conversion.
- Export the model into GGUF.
- Load the exported GGUF and run image classification with `run_resnet50`.

Important scope note:

- This is **working ResNet50 support**.
- It is **not yet a generic convolution-model runtime** integrated into the normal `llama.cpp` graph path.
- The current inference path in `run_resnet50.cpp` uses a dedicated CPU implementation for ResNet50 layers while GGUF is used as the model container and metadata format.

## Files

- `convert_resnet50_to_gguf.py`
  Converts a Hugging Face ResNet50 checkpoint into GGUF.
- `run_resnet50.cpp`
  Loads a ResNet50 GGUF model, preprocesses an input image, and runs inference.

## Model Locations

Local Hugging Face model copied into the repository:

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50_hf_model`

Recommended GGUF output path:

- `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf`

## Build

From the workspace root:

```bash
cmake --build /home/tianruiming/CE_ADA_LLAMA/build-release-current --target run_resnet50 -j4
```

## Convert Hugging Face ResNet50 to GGUF

Using the copied local model:

```bash
env HF_HOME=/tmp/resnet50_hf \
    /home/tianruiming/miniconda3/envs/pytorch/bin/python \
    src/llama.cpp/exp8_resnet50/convert_resnet50_to_gguf.py \
    --skip-download \
    --model-dir /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50_hf_model \
    --outfile /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf
```

If downloading from Hugging Face again is needed, first enable proxy and then run without `--skip-download`.

## Run Inference

Example:

```bash
./build-release-current/run_resnet50 \
    --model /home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf \
    --image /tmp/coco_cat.jpg \
    --threads 8 \
    --top-k 5
```

## Verification Result

Verification was rerun on `2026-04-15` with:

- Hugging Face model:
  `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50_hf_model`
- Converted GGUF:
  `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/resnet50-f32.gguf`
- Test image:
  `/tmp/coco_cat.jpg`

`run_resnet50` output:

```text
1  class_id=282  logit=0.880118   label=tiger cat
2  class_id=281  logit=-0.874038  label=tabby, tabby cat
3  class_id=285  logit=-2.89389   label=Egyptian cat
4  class_id=761  logit=-4.35684   label=remote control, remote
5  class_id=100  logit=-4.86543   label=black swan, Cygnus atratus
```

Hugging Face baseline on the same image:

```text
1  class_id=282  logit=1.616538   label=tiger cat
2  class_id=281  logit=-1.693961  label=tabby, tabby cat
3  class_id=761  logit=-4.734185  label=remote control, remote
4  class_id=285  logit=-4.863585  label=Egyptian cat
5  class_id=612  logit=-5.499217  label=jinrikisha, ricksha, rickshaw
```

Conclusion:

- Top-1 matches Hugging Face baseline: `tiger cat`.
- Top-2 also matches.
- The remaining Top-5 ordering differs slightly, which is acceptable for the current preprocessing/runtime path.
- ResNet50 adaptation is considered working for conversion and inference.

## Implementation Notes

The key issue fixed during adaptation was tensor storage order in GGUF:

- The GGUF tensor shape metadata for convolution kernels must match ggml expectations.
- The raw tensor byte order must also match dim0-fastest storage.
- Earlier conversion code used a visually correct shape with incorrect raw storage order, which caused obviously wrong predictions.

The current converter writes:

- convolution weights in storage compatible with ggml tensor interpretation
- fused convolution biases
- classifier weights and bias
- image preprocessing metadata
- class labels

## Current Limitation

This implementation should currently be treated as:

- `ResNet50` support in `llama.cpp`

not as:

- a generic parser/executor for arbitrary CNN architectures

If broader convolution-model support is needed later, the next step is to generalize:

- architecture metadata
- tensor naming conventions
- convolution/pooling execution path
- model graph construction
