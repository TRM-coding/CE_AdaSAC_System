# Mini-batch vs Real-data SVD Accuracy Experiment

This directory contains an independent experiment for comparing CE-AdaSAC-style
synthetic mini-batches with equal-sized real mini-batches under randomized SVD
pruning strategies.

## What the Script Does

- Loads the exp11 synthetic mini-batches:
  - `exp11_make_input_data/data/qwen2_5_1_5b_synthetic_minibatch.pt`
  - `exp11_make_input_data/data/resnet50_synthetic_minibatch.pt`
- Builds equal-sized real batches:
  - Qwen2.5-1.5B: same batch size and sequence length as the synthetic embedding batch, using a local text file.
  - ResNet50: same batch size as the synthetic image batch, sampled from `/SSD/val`.
- Randomizes several SVD pruning plans.
- Replaces selected modules with local low-rank factors:
  - `Linear`: `x -> x V_k -> U_k`
  - `Conv2d`: original convolution is decomposed into a rank-`k` spatial convolution followed by `1x1` convolution.
- Does not run cooperative/offload recovery for the discarded tail singular directions.

The x-axis in the generated plot is total SVD pruning amount, computed from the
randomized per-layer keep ratios. The y-axis is PPL for Qwen and top-1 ACC for
ResNet50. The ResNet50 panel is drawn as scatter points plus a light line
through points sorted by pruning amount.

## Run

```bash
cd /home/tianruiming/CE_ADA_LLAMA
/home/tianruiming/miniconda3/envs/pytorch/bin/python \
  src/llama.cpp/datas/acc_evaluate/svd_minibatch_vs_real.py \
  --out-dir src/llama.cpp/datas/acc_evaluate/results/minibatch_vs_real_svd_20260429
```

Useful knobs:

- `--num-policies`: number of randomized SVD plans, excluding the baseline.
- `--max-qwen-modules-per-policy`: Qwen MLP Linear modules randomly pruned per plan.
- `--max-resnet-modules-per-policy`: ResNet Conv/Linear modules randomly pruned per plan.
- `--keep-min`, `--keep-max`: per-selected-layer SVD keep-ratio range.
- `--tasks qwen` or `--tasks resnet50`: run one model only.
- To sweep ResNet50 across the full pruning range without rerunning Qwen:

```bash
/home/tianruiming/miniconda3/envs/pytorch/bin/python \
  src/llama.cpp/datas/acc_evaluate/svd_minibatch_vs_real.py \
  --tasks resnet50 \
  --num-policies 100 \
  --resnet-target-prune-sweep \
  --resnet-min-prune-percent 1 \
  --resnet-max-prune-percent 100 \
  --svd-device cuda:0 \
  --resnet-cache-svd-on-device \
  --out-dir src/llama.cpp/datas/acc_evaluate/results/resnet_target_sweep_100_20260429
```

## Current Result

Generated files:

- `results/minibatch_vs_real_svd_20260429/svd_minibatch_vs_real.csv`
- `results/minibatch_vs_real_svd_20260429/svd_minibatch_vs_real.png`
- `results/minibatch_vs_real_svd_20260429/args.json`
- `results/minibatch_vs_real_svd_20260429_resnet30/svd_minibatch_vs_real.csv`
- `results/minibatch_vs_real_svd_20260429_resnet30/svd_minibatch_vs_real.png`
- `results/minibatch_vs_real_svd_20260429_resnet30/resnet_args.json`
- `results/resnet_target_sweep_100_20260429/svd_minibatch_vs_real.csv`
- `results/resnet_target_sweep_100_20260429/svd_minibatch_vs_real.png`
- `results/minibatch_vs_real_svd_20260429_resnet100_fullrange/svd_minibatch_vs_real.csv`
- `results/minibatch_vs_real_svd_20260429_resnet100_fullrange/svd_minibatch_vs_real.png`

Run settings:

- 5 randomized pruning policies plus baseline.
- Extra ResNet50 scatter/line run: 30 randomized pruning policies plus baseline.
- Full-range ResNet50 run: 100 targeted randomized pruning policies plus baseline.
- Qwen2.5-1.5B: 4 synthetic embedding samples vs 4 real text samples, sequence length 16.
- ResNet50: 8 synthetic images vs 8 ImageNet validation images.
- Qwen randomized pruning: 6 MLP Linear modules per policy.
- ResNet randomized pruning: 10 Conv/Linear modules per policy.
- Full-range ResNet randomized pruning: all Conv/Linear modules are eligible; target total pruning sweeps from 1% to 100%.
- Full-range speed path: ResNet SVD factors are cached once on GPU, then each plan slices cached rank prefixes.

Observed ranges in this run:

- Qwen synthetic PPL: `1.0008 -> 1.0862`.
- Qwen real-text PPL: `15.8185 -> 33.9630`.
- ResNet synthetic ACC in the 30-policy run: `1.000 baseline`, then `0.000 -> 1.000` across randomized plans.
- ResNet real ImageNet ACC in the 30-policy run: `0.875 baseline`, then `0.000 -> 0.875` across randomized plans.
- ResNet full-range prune coverage: `0.000 baseline`, then `0.010 -> 0.997` actual pruning. The max is below `1.000` because each SVD module keeps at least rank 1.
- ResNet synthetic ACC in the 100-policy full-range run: `1.000 baseline`, then `0.000 -> 1.000`.
- ResNet real ImageNet ACC in the 100-policy full-range run: `0.875 baseline`, then `0.000 -> 0.875`.

The small batch sizes are intentional for this experiment, but they make ACC
coarse for ResNet50. Each correct/incorrect sample changes ACC by `0.125`.
