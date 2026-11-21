# DyT-ViT: Dynamic Tanh Vision Transformer on CIFAR-100

Reproduction of the core idea from *Transformers without Normalization* (Zhu et al., CVPR 2025, [arXiv:2503.10622](https://arxiv.org/abs/2503.10622)): replacing every `LayerNorm` in a Vision Transformer with the element-wise **Dynamic Tanh (DyT)** layer.

## References

- **Paper:** J. Zhu, X. Chen, K. He, Y. LeCun, Z. Liu. *Transformers without Normalization.* CVPR 2025.
- **Dataset:** A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* University of Toronto, 2009. (CIFAR-100)
- **Official code (reference only):** [github.com/jiachenzhu/DyT](https://github.com/jiachenzhu/DyT)

## Experiment Summary

Three models are trained under an identical recipe (seed, data split, optimizer, schedule, epochs). The only difference is the normalization layer:

| Model | Normalization | α₀ |
|---|---|---|
| A — LayerNorm-ViT | `nn.LayerNorm` | — |
| B — DyT-ViT | `DynamicTanh` | 0.5 |
| C — DyT-ViT-Var | `DynamicTanh` | 1.0 |

Model C is a controlled variation testing sensitivity to the initial α value.

## How to Run

### 1. Upload CIFAR-100 zip to Drive (once)

Zip so the archive contains:

```
cifar-100-python/
  train
  test
  meta
```

Upload to Google Drive as:

```
MyDrive/cifar-100-python.zip
```

(Change `DRIVE_CIFAR_ZIP` in the notebook if you use another path.)

### 2. Google Colab

1. Upload `transformer_dyt.ipynb` to Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Keep `DEBUG_SMOKE_TEST = False`.
4. Run all cells.
5. The notebook mounts Drive, copies the zip → `./data`, extracts it, and trains with `download=False` (no internet dataset download).
6. **All outputs** (checkpoints, histories, plots, tables) go to `MyDrive/DyT_ViT_CIFAR100_Runs/` so they survive disconnects.
7. Expect ~1 hour per model on a T4 (~3 hours total).

### Resuming after a Colab disconnect

1. Reconnect with the same GPU runtime.
2. Re-run setup through the training engine (imports → Drive mount → config → **prepare CIFAR zip** → data loaders → model classes → `run_experiment`).
3. Run only the interrupted model’s training cell — it resumes from `latest_checkpoint.pt` on Drive.

### Local (CPU — for testing only)

```bash
pip install torch torchvision scikit-learn pandas matplotlib
```

Use an extracted `./data/cifar-100-python/` (or a local zip). Set `DEBUG_SMOKE_TEST = True` for a 2-epoch sanity check. Outputs go to `./runs`.

## Outputs

On Colab these live under `MyDrive/DyT_ViT_CIFAR100_Runs/`:

```
├── comparison_table.csv
├── paper_vs_ours.csv
├── test_results.json
├── split_indices.json
├── ln_vit/          best_model.pt, latest_checkpoint.pt, history.json, test_results.json
├── dyt_vit_a05/     ...
├── dyt_vit_a10/     ...
└── plots/
    ├── training_curves.png/pdf
    ├── alpha_trajectories.png/pdf
    ├── coarse_confusion_matrices.png/pdf
    ├── top10_confused_pairs.png/pdf
    ├── cifar100_samples.png
    └── dyt_vs_layernorm_mapping.png
```
