# ELMo Mini Lab (TensorFlow)

A small, step-by-step Python project for **learning how the ELMo (Embeddings from Language Models) model works in practice** using **TensorFlow 2** and **TensorFlow Hub**.

The goal of this mini-lab is to:
- understand **what ELMo takes as input**,
- inspect **what vectors it produces as output**,
- explore **token-level vs sentence-level embeddings**,
- and clearly see **contextual embeddings in action** (e.g. the word *"bank"* in different contexts).

This project is intentionally minimal and educational — no training, no fine-tuning, just **inspection and intuition**.

---

## Project Structure

```
elmo-mini-lab/
│
├── requirements.txt
│
├── ### ELMo files
├── inspect_elmo.py # Inspect available ELMo outputs and tensor shapes
├── token_vs_sentence.py # Compare sentence-level and token-level embeddings
└── context_demo_bank.py # Contextual demo: "bank" in different meanings
```

---

## Requirements

- Python **3.9+**
- TensorFlow **2.10+**
- TensorFlow Hub

Install dependencies:

```bash
pip install -r requirements.txt
```

## Step-by-Step Overview

1. Inspect the ELMo model
File: inspect_elmo.py

This script:

- loads ELMo from TensorFlow Hub,

- prints available SavedModel signatures,

- shows all output tensors, their names, shapes and data types.

Purpose:

Understand what exactly comes out of ELMo before using it.

Run:
```bash
python inspect_elmo.py
```


2. Sentence vs Token embeddings
File: token_vs_sentence.py

This script:

- detects 2D outputs (sentence-level embeddings),

- detects 3D outputs (token-level embeddings),

- prints example vectors and their norms.

Purpose:

See the structural difference between pooled sentence embeddings and per-token embeddings.

Run:

```bash
python token_vs_sentence.py
```

3. Contextual meaning demo: bank
File: context_demo_bank.py

This script compares the word "bank" in two sentences:

- financial context

- river context

It:

- extracts token-level embeddings,

- finds the embedding of "bank" in each sentence,

- computes cosine similarity between them.

Purpose:

Demonstrate contextual embeddings in the most intuitive way.

Run:

```bash
python context_demo_bank.py
```


Expected result:

- cosine similarity significantly lower than 1

- showing that ELMo encodes meaning, not just spelling

Output embeddings are context-dependent

Token embeddings have shape:
```
[batch_size, sequence_length, embedding_dim]
```

Sentence embeddings have shape:
```
[batch_size, embedding_dim]
```



## References
Peters et al., Deep contextualized word representations, 2018

TensorFlow Hub: https://tfhub.dev/google/elmo/3

Original paper: https://arxiv.org/abs/1802.05365