import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

HANDLE = "https://tfhub.dev/google/elmo/3"

def cosine(a, b, eps=1e-9):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))

def main():
    elmo = hub.load(HANDLE)
    sig_name = "default" if "default" in elmo.signatures else list(elmo.signatures.keys())[0]
    f = elmo.signatures[sig_name]

    # s1 = "I went to the bank to deposit money"
    # s2 = "I sat on the bank of the river"
    s1 = "The central bank raised interest rates yesterday."
    s2 = "A fisherman rested on the river bank at sunset."

    batch = tf.constant([s1, s2])
    out = f(batch)

    # tensor 3D (token embeddings)
    token_key = None
    token_emb = None
    for k, v in out.items():
        if "lstm_outputs2" == k:
            token_key = k
            token_emb = v.numpy()  # [B, T, 1024]
            break
        elif len(v.shape) == 3:
            token_key = k
            token_emb = v.numpy()  # [B, T, 1024]

    if token_emb is None:
        print("Error. 3D token-embeddings in output not found. Run inspect_elmo.py and get 3D tensor key.")
        return

    # tokenize
    t1 = s1.split()
    t2 = s2.split()

    # find token bank
    def find_bank(tokens):
        for i, tok in enumerate(tokens):
            if tok.lower().strip(".,;:!?") == "bank":
                return i
        return None

    i1 = find_bank(t1)
    i2 = find_bank(t2)

    print("Token key used:", token_key)
    print("Tokens 1:", t1)
    print("Tokens 2:", t2)
    print("Index bank:", i1, i2)

    if i1 is None or i2 is None:
        print("Token 'bank' not found after split().")
        return

    v1 = token_emb[0, i1]  # [1024]
    v2 = token_emb[1, i2]  # [1024]

    print("Cosine similarity(bank in finance vs bank in river) =", cosine(v1, v2))

    print("---------------------")

    layers = ["word_emb", "lstm_outputs1", "lstm_outputs2", "elmo"]
    for layer in layers:
        v1 = out[layer].numpy()[0, i1]
        v2 = out[layer].numpy()[1, i2]
        print(layer, cosine(v1, v2))

if __name__ == "__main__":
    main()
