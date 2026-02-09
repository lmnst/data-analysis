## README — Dataset Analysis & Recommended Usage

### Overview

This repository uses multiple datasets for a humor generation pipeline. We performed lightweight EDA (data validation + quality checks) to ensure the datasets are suitable for:

1. **RAG retrieval corpus**, and
2. **Supervised fine-tuning (SFT)** for LLaMA-style chat models.

---

## 1) `hahackathon_train.csv` (Humor Scoring / Evaluator)

### Dataset structure and label coverage

- Total rows: **8,000**
- Fully labeled / trainable for humor scoring: **4,932**
- Missing `humor_rating` and `humor_controversy`: **3,068 (38.35%)**
- Missingness is *systematic*, not random:
  - All rows **with** humor ratings have `is_humor = 1`
  - All rows **without** humor ratings have `is_humor = 0`

This indicates the file is effectively a combination of:

- a **rated humor subset** (`is_humor=1`) used to train a humor scorer, and
- a **non-humor negative subset** (`is_humor=0`) useful for training a humor detector.

### Humor rating distribution (rated subset, n=4,932)

- `humor_rating` range: **0.1–4.0**
- mean: **2.26**, std: **0.57**
- Rounded counts:
  - 0: 16
  - 1: 410
  - 2: 2,835
  - 3: 1,624
  - 4: 47
  - (no 5s)

This distribution is highly concentrated around **2–3**, and high-score examples are scarce.

### Correlations (rated subset)

Computed on the rated subset (n=4,932).

- Humor vs Offense (Pearson): **-0.309**
- Humor vs Controversy (Pearson): **+0.174**

“High humor & low offense” (`humor ≥ 3.5` and `offense ≤ 1.5`) exists but is rare:

- **45 samples (0.91%)**

### Recommended training strategy

Given the dataset structure, we recommend:

- **Two-stage pipeline**:
  1. Train a binary humor detector on all 8,000 samples (`is_humor`)
  2. Train a humor scorer only on the rated subset (4,932 samples)
- Or a **multi-head model** with **masked loss** for `humor_rating`/`humor_controversy` when ratings are missing.

Exported training files:

- `train_detector.csv` (8,000 rows)
- `train_scorer.csv` (4,932 rows)
- `train_multitask.csv` (8,000 rows, with `has_humor_score` for loss masking)

---

## 2) JSONL Datasets for RAG and SFT

### JSONL format

Both JSONL datasets follow a ChatML-style schema:

```json
{
  "messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
  "word1": "...",
  "word2": "..."
}
```

Validation:

- Parsing errors: **0**
- Missing required fields: **0**
- Word constraint compliance: **100%** (assistant output contains both `word1` and `word2`)

------

### Dataset A: `humor_RAG_data_20000_RAG.jsonl` (RAG corpus)

**Before dedup**

- Rows: **20,000**
- Answer duplication rate: **0.172**
- Pair duplication rate (prompt+answer): **0.0**

**After dedup by assistant answer**

- Output: `humor_RAG_data_20000_RAG_dedupByAnswer.jsonl`
- Rows: **16,557** (kept **82.78%**, dropped 3,443)
- Answer duplication rate: **0.0**
- Word constraint pass rate: **1.0**
- Median / P90 answer length (chars): **114 / 142**

**Recommended usage**

- Use the **deduplicated** file as the primary **RAG retrieval corpus** to maximize diversity.

------

### Dataset B: `humor_training_data_5000_Train.jsonl` (SFT training)

**Before dedup**

- Rows: **5,000**
- Answer duplication rate: **0.375**
- Pair duplication rate (prompt+answer): **0.0**

**After dedup by assistant answer**

- Output: `humor_training_data_5000_Train_dedupByAnswer.jsonl`
- Rows: **3,121** (kept **62.42%**, dropped 1,879)
- Answer duplication rate: **0.0**
- Word constraint pass rate: **1.0**
- Median / P90 answer length (chars): **110 / 148**

**Recommended usage**

- Use the **deduplicated** file for **SFT fine-tuning** to reduce repetition and improve generalization.

------

### Scripts

- `scripts/analyze_jsonl_humor.py`: format validation + constraint pass rate + duplication and length stats
- `scripts/dedup_jsonl_by_answer.py`: deduplicate by normalized assistant answer text and export clean JSONL files

------

## What’s next (baseline training)

- Humor detection (binary): train on `train_detector.csv`
- Humor scoring: train on `train_scorer.csv` using:
  - bucketed classification (low/mid/high) **and** regression baseline (MAE + Spearman)
- Optional: multi-head model with masked loss using `train_multitask.csv