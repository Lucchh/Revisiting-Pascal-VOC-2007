# Revisiting Pascal VOC 2007

Controlled research framework for comparing U-Net, DeepLabV3, and SAM2-adapter segmentation architectures on the Pascal VOC 2007 benchmark. The codebase mirrors the methodology in *“Revisiting Pascal VOC 2007: A Comparative Study of U-Net, DeepLabV3, and SAM2Adapter for Semantic Segmentation”* and contains stage-aligned training, evaluation, and visualization utilities for publication-quality studies.

---

## Repository Layout

| Path | Description |
| --- | --- |
| `configs/` | YAML experiment definitions (data paths, optimization, model registry, augmentation knobs). |
| `scripts/train.py` | Main training loop with multi-model orchestration, checkpointing, and CSV logging. |
| `scripts/evaluate.py` | Offline evaluation on stored checkpoints; outputs aggregate metrics + per-class tables. |
| `scripts/visualize.py` | Generates qualitative grids/mosaics for validation images. |
| `scripts/run_ablation.py` | Convenience entry point for ablation sweeps (encoder depth, augmentation strength, loss). |
| `src/segmentation/` | Reusable library (data module, metrics, models, SAM2 adapters, utility functions). |
| `notebooks/` | Exploratory analysis (debug/test) and figure generation for the accompanying paper. |
| `reports/` | Publication template / notes. Large PDFs or figures should be staged here. |

All heavyweight artifacts (VOC datasets, SAM2 checkpoints, trained weights, tensorboard logs, `.venv`) are ignored via `.gitignore` so the repository only tracks the research-critical source.

---

## Environment Setup

```bash
conda create -n voc_seg python=3.10 -y
conda activate voc_seg
pip install -r requirements.txt

# optional: install SAM2 codebase for adapter experiments
git clone https://github.com/facebookresearch/sam2.git external/sam2
pip install -e external/sam2
```

> The SAM2 adapter expects checkpoints inside `external/sam2/`. Either run Meta’s `download_ckpts.sh` script or place the pre-trained `sam2.1_hiera_*.pt` files there manually.

---

## Data

1. Download the Pascal VOC 2007 train/val split (`VOCtrainval_06-Nov-2007`) and test split if needed for qualitative demos.
2. Place the extracted folder at the repository root (already referenced in `configs/experiment.yaml` by default).
3. No additional preprocessing is required; the dataloader handles resizing, ignore-index pixels (255), and normalization.

If you store the dataset elsewhere, update `voc_root` inside `configs/experiment.yaml` or pass `--voc-root /path/to/VOCdevkit`.

---

## Running Experiments

### Configure

Edit `configs/experiment.yaml` to choose:

- `models`: list of architectures to train (`unet_resnet{18,34,50}`, `deeplabv3_resnet50`, `sam_vit_b`, `sam2_base_plus`).
- `augmentations`: enable geometric/photometric transforms (`backend: torchvision` or `albumentations`).
- `optimizer`, `lr_scheduler`, and loss mix (CE vs CE+Dice) per the ablation knobs in the paper.

### Train

```bash
PYTHONPATH=src python scripts/train.py --config configs/experiment.yaml
```

Common overrides:

```bash
# restrict to two models and shorten the schedule
PYTHONPATH=src python scripts/train.py \
  --config configs/experiment.yaml \
  --models unet_resnet50 sam2_base_plus \
  --max-epochs 10
```

Checkpoints, metrics CSVs, and qualitative samples are saved to `runs/<experiment_name>/`.

### Evaluate

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/experiment.yaml \
  --checkpoint runs/voc2007_baseline/unet_resnet50/best.ckpt
```

The evaluator reports mIoU, Dice, pixel accuracy, HD95, and dumps per-class metrics for reproducibility.

### Visualize

```bash
PYTHONPATH=src python scripts/visualize.py \
  --config configs/experiment.yaml \
  --checkpoint runs/voc2007_baseline/sam2_base_plus/best.ckpt \
  --num-images 8
```

Use the resulting mosaics inside `figures/` or directly in LaTeX manuscripts.

---

## Reproducing
Steps:

1. Train each model with the default config (30 epochs, AdamW, 512×512 resizing, ignore-index 255).
2. Use `scripts/evaluate.py` to compute metrics on the VOC 2007 validation split.
3. Generate mosaics + best/worst case visualizations using `scripts/visualize.py` for inclusion in the manuscript.

Ablation scripts (`scripts/run_ablation.py`) toggle encoder depth, augmentation strength, and CE vs CE+Dice loss, matching the tables in the paper’s Appendix.

---

## Notebooks & Reporting

- `notebooks/debug.ipynb`, `notebooks/test.ipynb`: quick sanity checks and qualitative inspection.
- `reports/`: contains the LaTeX/Markdown templates used for the final write-up. Export plots from `figures/` or `results/` to keep the repository organized.

---

## Citation

If you use this scaffold or reproduce the study, please cite:

```
@article{chen2025revisitingvoc,
  title   = {Revisiting Pascal VOC 2007: A Comparative Study of U-Net, DeepLabV3, and SAM2Adapter for Semantic Segmentation},
  author  = {Luc Chen},
  journal = {Technical Report},
  year    = {2025}
}
```

---

For questions or collaboration requests, open an issue or reach out via the project repository. All contributions aligning with the experimental protocol are welcome.
