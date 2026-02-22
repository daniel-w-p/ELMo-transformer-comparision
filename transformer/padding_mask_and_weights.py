import numpy as np
import tensorflow as tf

from mini_transformer import MiniTransformer

sentences = [
    "Yesterday I went to the bank",
    "The river bank was wide"
]

tokenized = [s.split() for s in sentences]
vocab = sorted(set(tok for sent in tokenized for tok in sent))
stoi = {w: i+1 for i, w in enumerate(vocab)}

max_len = max(len(s) for s in tokenized)

def encode(sent):
    ids = [stoi[w] for w in sent]
    padding = [0]*(max_len - len(ids))
    return ids + padding

X = np.array([encode(s) for s in tokenized])

print("Encoded with padding:\n", X)

# Build attention pad mask
# mask shape: [batch, seq_len]; 1 for real, 0 for padding
pad_mask = (X != 0).astype("int32")

print("Pad mask:\n", pad_mask)

# Convert to transformer mask format (add dims)
# tf.keras.layers.MultiHeadAttention expects attention_mask in this shape:
# [batch, seq_len] or [batch, seq_len, seq_len]
mask = pad_mask[:, tf.newaxis, :]  # broadcast to [batch, 1, seq_len]

model = MiniTransformer(vocab_size=len(stoi)+1, max_len=max_len)

# Forward with attention
out, attn_scores = model(X, mask=mask, return_attention=True)

print("Output shape:", out.shape)
print("Attention scores shape:", attn_scores.shape)

# For easier inspection
print("Attention weights (first head, first example) [batch, head]:")
print(attn_scores[0, 0])
print(attn_scores[1, 0])