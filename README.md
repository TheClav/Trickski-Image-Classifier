# Trickski Angle Classifier

A computer vision model that classifies the body orientation of a trick water skier — the first building block toward a fully automated trick skiing judge.

## Background & Motivation

Competitive trick water skiing is a sport where athletes perform spins and tricks on a smooth, finless ski within a 20-second window, combining unique tricks to maximize their point total. Tournament runs are currently scored by human judges watching live — a system that is old, inconsistent, and prone to error.

As a collegiate water skier at UW-Madison, I wanted to fix that. The end goal is a system that can take a 20-second tournament video and automatically classify and score every trick in the run. This project is **Phase 1**: teaching a model to identify which direction the skier is facing.

## The Problem

A trick (e.g., a 360) is really just a sequence of orientations over time:

```
forward → left → back → right → forward  ≈  360 spin
```

So the first step is simple: **given a single image of a trick skier, classify their orientation as one of four classes:**

| Class | Description |
|---|---|
| `forward` | Skier facing toward the boat (normal riding position) |
| `left` | Skier turned 90° to the left |
| `right` | Skier turned 90° to the right |
| `back` | Skier facing away from the boat (back to boat) |

Once each frame of a trick clip is classified, the sequence of orientations can be matched to a known trick pattern to determine what trick was performed and assign its point value.

## Dataset

There was no existing dataset for trick skiing, so I built one from scratch.

- Reached out to the **waterski results** database to access archived tournament videos
- Manually reviewed, downloaded, edited, and labeled ~60 videos
- Extracted still frames and sorted them into orientation classes
- **Final dataset: ~240 images** (180 train / 60 test across 4 classes)

```
trickski_data/
├── train/
│   ├── train_forward/
│   ├── train_left/
│   ├── train_right/
│   └── train_back/
└── test/
    ├── test_forward/
    ├── test_left/
    ├── test_right/
    └── test_back/
```

## Model & Approach

### What didn't work

| Approach | Test Accuracy |
|---|---|
| Custom SimpleCNN (from scratch) | ~6% |
| SimpleCNN + hyperparameter tuning | ~10% |
| SimpleCNN + data augmentation (flips, rotation) | ~20% |
| Pretrained ResNet-18 (no changes) | ~50% |

The custom CNN never learned anything meaningful despite decreasing loss — a symptom of having far too little data. Switching to a **pretrained ResNet-18 fine-tuned on this dataset** jumped accuracy to ~50% immediately.

### Two key breakthroughs

1. **Data augmentation was actively hurting performance.** Flipping and rotating images destroyed orientation information — the exact thing the model was supposed to learn. Removing all augmentation was a significant win.

2. **Background noise was a major source of error.** Tournament footage has a lot of sky, water, and scenery. Adding a **center crop** to focus on the skier (who is almost always in the center of the frame) reduced noise and improved accuracy considerably.

### Final architecture

- **Model**: ResNet-18 pretrained on ImageNet, final FC layer replaced for 4-class output
- **Input**: 224×224 center-cropped images, normalized to ImageNet stats
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: CrossEntropyLoss
- **Epochs**: 40

### Results

**~81–85% test accuracy** on 60 held-out images.

## Project Structure

```
Trickski-Image-Classifier/
├── notebooks/
│   └── testing.ipynb       # Full training & evaluation pipeline
├── src/
│   ├── dataset.py          # Custom TrickSkiDataset (PyTorch Dataset)
│   └── model.py            # SimpleCNN architecture (experimental baseline)
├── trickski_data/          # Train/test images organized by class
└── 539 Final Project Presentation.pdf
```

## Setup & Usage

**Requirements:** Python 3.8+, PyTorch, torchvision, Pillow

```bash
pip install torch torchvision pillow
```

Open `notebooks/testing.ipynb` and run all cells. The notebook handles:
- Dataset loading via `TrickSkiDataset`
- Transform pipeline (resize → center crop → normalize)
- Fine-tuning ResNet-18
- Epoch-by-epoch loss and accuracy reporting

The notebook auto-detects CUDA, Apple MPS, or CPU.

## Key Lessons

- **Transfer learning is essential with small datasets.** A pretrained model gave an immediate 40-point accuracy boost over training from scratch.
- **Data augmentation can hurt orientation-sensitive tasks.** Flipping or rotating an image changes its label — don't augment when the augmentation destroys the signal you're trying to learn.
- **Reduce noise, focus on the subject.** A simple center crop outperformed more complex preprocessing by keeping the model's attention on the skier.

## Future Work

- **Dynamic skier cropping** — use the ski rope (a straight line from the center of the frame) to locate and crop the skier precisely, regardless of their position in the frame
- **Trick identification** — classify orientation for each frame in a short clip, then match the orientation sequence against known trick patterns (e.g., `F → L → B → R → F` = 360)
- **Automatic scoring** — assign and accumulate point values for each identified trick
- **End goal** — upload a 20-second tournament video and get a fully scored trickski run, start to finish

## Author

Calvin DeBellis — UW-Madison Collegiate Water Ski Team
