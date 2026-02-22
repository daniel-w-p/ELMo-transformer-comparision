import numpy as np

from mini_transformer import MiniTransformer

# simple whitespace tokenizer
sentences = [
    # "I went to the bank to deposit money",
    # "I sat on the bank of the river"
    "The central bank raised interest rates yesterday.",
    "A fisherman rested on the river bank at sunset."
]

tokenized = [s.split() for s in sentences]
print("Tokenized:", tokenized)

# Mini dictionary
vocab = sorted(set(tok for sent in tokenized for tok in sent))
stoi = {w: i+1 for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

max_len = max(len(s) for s in tokenized)

def encode(sent):
    ids = [stoi[w] for w in sent]
    return ids + [0]*(max_len - len(ids))

X = np.array([encode(s) for s in tokenized])

print("Encoded input:\n", X)

# Instantiate model
model = MiniTransformer(vocab_size=len(stoi)+1, max_len=max_len)

# Forward pass
out = model(X).numpy()  # [batch, seq_len, embed_dim]
print("Output shape:", out.shape)

# Cosine similarity
def cosine(a, b, eps=1e-9):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))

bank_idx = tokenized[0].index("bank")

v1 = out[0, bank_idx]
v2 = out[1, bank_idx]
print("Cosine similarity (mini-Transformer) of 'bank':", cosine(v1, v2))