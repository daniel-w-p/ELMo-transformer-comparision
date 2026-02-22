# ELMo vs BERT Mini Lab (TensorFlow)

A small, step-by-step Python project for **understanding contextual word embeddings in practice**, using **ELMo and BERT** with **TensorFlow 2** and **TensorFlow Hub**.

The goal of this mini-lab is **not training or fine-tuning**, but **inspection, comparison and intuition**.

We focus on:
- what **ELMo and BERT take as input**,
- what **exact tensors they output**,
- how **token-level and sentence-level embeddings differ**,
- how **contextual meaning emerges**,
- and how **ELMo and BERT differ conceptually and practically**.

This repository is designed as a **hands-on notebook replacement** — everything is explicit, printed, and measurable.

---

## Learning Goals

By the end of this mini-lab you should clearly understand:

- why contextual embeddings ≠ static word vectors,
- how **ELMo** and **BERT** represent tokens internally,
- how **subword tokenization** (BERT) changes embedding inspection,
- what “sentence embedding” actually means in each model,
- why the same word has different vectors in different contexts.

---

## Project Structure

```
elmo-mini-lab/
│
├── requirements.txt
│
├── ### ELMo files
├── elmo
│   ├── inspect_elmo.py # Inspect available ELMo outputs and tensor shapes
│   ├── token_vs_sentence.py # Compare sentence-level and token-level embeddings
│   └── context_demo_bank.py # Contextual demo: "bank" in different meanings
│
├── ### Mini-Transformer
├── transformer
│   ├── mini_transformer.py # A simple Transformer model for testing embeddings
│   ├── inspect_transformer.py # Inspect mini transformer outputs and tensor shapes
│   └── padding_mask_and_weights.py # Inspect padding masking and attention weights
│
├── ### BERT files
└── bert
    └── inspect_bert.py # Inspect BERT outputs and tensor shapes
```

---

## Requirements

- Python **3.9+**
- TensorFlow **2.10+**
- TensorFlow Hub
- TensorFlow Text
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

# Step-by-Step Overview

## Part 1. ELMo model

### 1. Inspect the ELMo model
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


### 2. Sentence vs Token embeddings
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

### 3. Contextual meaning demo: bank
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

#### Expected result:

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

## Part 2: BERT — contextual subword embeddings

Unlike ELMo, BERT:

uses WordPiece tokenization,

produces embeddings per layer,

represents sentences via a special [CLS] token.

This section mirrors the ELMo experiments step by step.

### 4. Inspecting the BERT model

File: inspect_bert.py

This script:

loads BERT from TensorFlow Hub,

prints all available outputs,

inspects shapes such as:

[batch_size, seq_len, hidden_dim]

Purpose:
Understand what BERT outputs before interpreting them.

---

## References & Licenses

### Models
*   **ELMo**: Peters et al., [Deep contextualized word representations](https://arxiv.org/abs/1802.05365), 2018. Source: [TF Hub](https://tfhub.dev/google/elmo/3). License: [Apache 2.0](https://www.apache.org).
*   **BERT**: Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), 2018. Source: [TF Hub](https://www.kaggle.com/models/tensorflow/bert/tensorFlow2/bert-en-uncased-l-12-h-768-a-12). License: [Apache 2.0](https://www.apache.org).

### License for the project
My original code in this repository is licensed under the **MIT License**.
