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

### Google Colab (recommended)

1. Upload `transformer_dyt.ipynb` to Google Colab.
2. Set the runtime to **GPU** (T4 or better): `Runtime → Change runtime type → T4 GPU`.
3. Set `DEBUG_SMOKE_TEST = False` in the config cell (this is the default).
4. **Run All** (`Runtime → Run all`).
5. CIFAR-100 downloads automatically. Checkpoints are saved to Google Drive under `MyDrive/DyT_ViT_CIFAR100_Runs/`.
6. Each model takes roughly **1 hour** on a T4 (≈ 3 hours total).

### Resuming after a Colab disconnect

1. Reconnect and select the same GPU runtime.
2. Run all cells **up to and including** the data-loading cell.
3. Run the training cell for the model that was interrupted — it auto-resumes from the last completed epoch via `latest_checkpoint.pt`.

### Local (CPU — for testing only)

```bash
pip install torch torchvision scikit-learn pandas matplotlib
```

Set `DEBUG_SMOKE_TEST = True` in the config cell to run a 2-epoch sanity check on a small data subset.

## Outputs

After a full run, `SAVE_DIR` contains:

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
