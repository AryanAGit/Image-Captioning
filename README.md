# Image Captioning — Notebook Guide

This project trains and evaluates image-captioning models on **MS-COCO** (an image
encoder feeding a text decoder). The work is split across several notebooks: some
**build/train** models, others **load a trained checkpoint and analyze it**.

Every notebook shares the same conventions, so checkpoints are interchangeable:

- **Data:** COCO `val2017`, split **90% train / 10% val by image id** (seed 42, so the
  same images always land in the same split — no leakage). No separate test set; the
  held-out val 10% is used both for model selection and reporting.
- **Checkpoint format:** a single self-contained `.pt` saved with
  `torch.save({"state_dict", "config", "bleu4", "epoch"}, path)`. The stored `config`
  lets any notebook rebuild the exact architecture. (No `.yml` is needed.)
- **Metric:** BLEU-4 (corpus + sentence level) via NLTK with `SmoothingFunction().method1`
  and lowercased `word_tokenize`, scored against **all** human references per image.
- **Architectures:** an `ImageEncoder` (CNN/ViT/CLIP) → a decoder (`GPT2Decoder` with
  cross-attention, or a word-level `GRUDecoder`), wrapped in a `CaptioningModel`.

---

## How the notebooks fit together

```
          EXPLORATION / TRAINING                         EVALUATION / ANALYSIS
  ┌──────────────────────────────────┐        ┌─────────────────────────────────────┐
  │ Image_Captioning (4)             │        │ Image_Captioning_Model_Eval         │
  │   CNN + LSTM/GRU baseline        │        │   score all images, rank, keywords  │
  │                                  │        │                                     │
  │ Image_Captioning_Transformers    │        │ Image_Captioning_Beam_Compare       │
  │   add ViT/CLIP + GPT-2 framework │  .pt   │   BLEU vs beam size                 │
  │                                  │ ─────► │                                     │
  │ Image_Captioning_Finetune_Compare│        │ Image_Captioning_TopP_Sweep         │
  │   tune + compare CNN/ViT/CLIP    │        │   BLEU vs nucleus top-p             │
  │                                  │        │                                     │
  │ Image_Captioning_ViT_GPT2        │        │ Image_Captioning_Infer              │
  │   train the chosen best config   │        │   caption one image, 3 strategies   │
  └──────────────────────────────────┘        └─────────────────────────────────────┘
```

The training notebooks **produce** `.pt` checkpoints; the evaluation notebooks **consume**
one via a `MODEL_PATH` setting. All of them run on Google Colab (auto-mount Drive,
auto-download COCO) and also locally.

---

## Training / exploration notebooks

### `Image_Captioning (4).ipynb` — CNN + RNN baseline (the from-scratch model)
The original "submission" notebook. Builds a classic **Show-and-Tell-style** captioner
from scratch:
- **Encoder:** `EncoderCNN` — a frozen ResNet with the classifier head removed, projected
  to an embedding vector.
- **Decoder:** `DecoderRNN` — a word-level **LSTM or GRU** (selectable) over a vocabulary
  built from training captions only (`freq_threshold` → `<unk>`).
- Does its own COCO download, image-level split, vocabulary, `Dataset`/`DataLoader`,
  training loop, BLEU evaluation, and checkpointing.
- **Section 10–11: randomized grid search** over hyperparameters, then retrains the best
  config. Ends with qualitative examples and a demo function.

Think of this as the **baseline / starting point** before transformers were introduced.

### `Image_Captioning_Transformers.ipynb` — architecture comparison framework
Introduces the **unified encoder/decoder framework** the rest of the project relies on:
- A single `ImageEncoder(kind, name)` that can be a **CNN (ResNet-50)**, a **ViT**, or
  **CLIP** vision tower — each returning a sequence of feature vectors `(B, S, D)`.
- Two interchangeable decoders: a word-level **GRU** and **GPT-2 with cross-attention**.
- **Section 7: architecture search** — trains several encoder/decoder combinations and
  compares their BLEU so you can see which pairing works best.
- Saves the best model to Drive; ends with qualitative examples and report notes.

This is where the project moves from "one hand-built CNN-RNN" to "a configurable
framework that can mix CNN/ViT/CLIP with GRU/GPT-2."

### `Image_Captioning_Finetune_Compare.ipynb` — tune & compare the top architectures
Takes the framework and runs a **two-phase study** on the strongest candidates
(CNN+GPT-2, ViT+GPT-2, CLIP+GPT-2, plus a CNN+GRU baseline):
- **Phase 1 (Section 8):** hyperparameter search per architecture.
- **Phase 2 (Section 9):** full training (15 epochs + early stopping) of each.
- **Section 10:** a single comparison plot of all models' BLEU curves.
- Saves the overall best checkpoint to Drive.

This is the **main experiment notebook** — it answers "which encoder pairs best with GPT-2,
and what does the BLEU comparison look like." It's also the source of the canonical
framework code (encoder/decoder classes, `ExperimentConfig`, checkpoint format) reused by
the evaluation notebooks.

### `Image_Captioning_ViT_GPT2.ipynb` — train the chosen best config at full scale
Once ViT+GPT-2 was selected as the winner, this notebook trains **just that one config**
on the full training set (15 epochs, early stopping) and saves `vit_gpt2_full.pt` to Drive.
- Streamlined to a single architecture (no search), with visualizations and report notes.
- **Note:** its `ExperimentConfig` is ViT-specific and does **not** store an
  `encoder_kind` field. The evaluation notebooks detect this and infer the field from the
  encoder name, so its checkpoints still load everywhere.

---

## Evaluation / analysis notebooks (consume a trained `.pt`)

All four take a `MODEL_PATH` pointing at a checkpoint, rebuild the model from its stored
`config`, and run a specific analysis. They include a **robust loader** that fills in any
config fields older checkpoints (like `vit_gpt2_full.pt`) omit.

### `Image_Captioning_Model_Eval.ipynb` — full-dataset scoring + error analysis
Loads one checkpoint and scores **every unique image** with sentence-level BLEU-4, then:
- Ranks all images by score and shows the **lowest-K and highest-K** examples
  (image + predicted caption + references + score grids).
- Compares the **most common keywords** in the model's captions for the worst vs. best
  groups (top-J content words, stopword-filtered, bar charts + set difference) — a quick
  way to see *what kinds of images the model fails on*.
- Saves results to Drive. Uses **greedy** decoding.

Use this to understand **where and how a model fails**, not just its average score.

### `Image_Captioning_Beam_Compare.ipynb` — BLEU vs. beam size
Sweeps **beam search width** (`[1, 2, 3, 5, …]`, where beam 1 == greedy) and reports
corpus/mean BLEU-4, average caption length, and runtime for each.
- Hand-written beam search for both GPT-2 and GRU decoders (HF `generate` can't forward
  the image cross-attention, so beam search is implemented from scratch).
- Plots BLEU vs. beam size and shows how a caption changes as the beam widens.
- Extra cells analyze **repetition** and the **lowest-greedy-BLEU** images (greedy vs.
  beam side by side).

Use this to pick a decoding width and to study greedy repetition vs. the beam-search
length bias.

### `Image_Captioning_TopP_Sweep.ipynb` — BLEU vs. nucleus (top-p) sampling
The sampling counterpart to the beam notebook. Decodes the eval images with **nucleus
(top-p) sampling** at several `top_p` values and reports corpus/mean BLEU-4 for each,
plus a deterministic **greedy baseline** for reference.
- Each `top_p` is averaged over several stochastic draws (mean ± std on the plot).
- Shows the **diversity-vs-BLEU trade-off**: BLEU falls as `top_p` rises (more varied
  captions, less reference overlap). Includes a qualitative grid of captions per `top_p`.

Use this to choose a sampling setting when you care about caption diversity / avoiding
repetition rather than maximizing BLEU.

### `Image_Captioning_Infer.ipynb` — caption a single image (demo)
Point it at **one image** and **one checkpoint**, and it generates a caption.
- No COCO download for GPT-2 models (only fetches annotations to rebuild the word
  vocabulary if the checkpoint is a CNN+GRU model).
- **Section 8: compares decoding strategies** on the same image — **greedy vs. beam vs.
  nucleus** (several nucleus draws), and can report **BLEU-4 per strategy** if you paste
  the image's ground-truth caption(s) into `REFERENCE_CAPTIONS`.
- Built for quick demos and qualitative checks on your own pictures.

---

## Quick reference

| Notebook | Role | Decoding | Produces / Consumes |
|---|---|---|---|
| `Image_Captioning (4)` | CNN+LSTM/GRU baseline, grid search | greedy | produces `.pt` |
| `Image_Captioning_Transformers` | CNN/ViT/CLIP + GRU/GPT-2 framework + arch search | greedy | produces `.pt` |
| `Image_Captioning_Finetune_Compare` | tune + compare top architectures | greedy | produces `.pt` (canonical framework) |
| `Image_Captioning_ViT_GPT2` | train chosen best (ViT+GPT-2) full scale | greedy | produces `vit_gpt2_full.pt` |
| `Image_Captioning_Model_Eval` | score all images, rank, keyword error analysis | greedy | consumes `.pt` |
| `Image_Captioning_Beam_Compare` | BLEU vs beam size, repetition analysis | beam | consumes `.pt` |
| `Image_Captioning_TopP_Sweep` | BLEU vs nucleus top-p, diversity trade-off | nucleus | consumes `.pt` |
| `Image_Captioning_Infer` | caption one image, compare 3 strategies | greedy/beam/nucleus | consumes `.pt` |

**Typical workflow:** train with one of the training notebooks → copy the resulting `.pt`
to Drive → set `MODEL_PATH` in an evaluation notebook → analyze (full-dataset scoring,
decoding sweeps, or single-image demos).
