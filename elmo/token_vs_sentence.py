import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

HANDLE = "https://www.kaggle.com/models/google/elmo/TensorFlow1/elmo/3"

def pick_outputs(outputs: dict):
    """Heuristics: tensor 2D (pooled) and 3D (token-level)."""
    pooled = None
    tokens = None
    for k, v in outputs.items():
        if len(v.shape) == 2 and pooled is None:
            pooled = (k, v)
        if len(v.shape) == 3 and tokens is None:
            tokens = (k, v)
    return pooled, tokens

def main():
    elmo = hub.load(HANDLE)
    sig_name = "default" if "default" in elmo.signatures else list(elmo.signatures.keys())[0]
    f = elmo.signatures[sig_name]

    batch = tf.constant([
        "ELMo gives contextual embeddings",
        "ELMo embeddings depend on context"
    ])
    out = f(batch)

    pooled, tokens = pick_outputs(out)

    print("All keys:", list(out.keys()))
    if pooled:
        k, v = pooled
        print(f"\nPooled candidate: {k} shape={v.shape} dtype={v.dtype}")
        v_np = v.numpy()
        print("Example pooled vector[0][:8] =", np.round(v_np[0][:8], 4))
        print("Norm pooled[0] =", float(np.linalg.norm(v_np[0])))

    if tokens:
        k, v = tokens
        print(f"\nToken-level candidate: {k} shape={v.shape} dtype={v.dtype}")
        v_np = v.numpy()
        print("Example token vector[0,0][:8] =", np.round(v_np[0, 0][:8], 4))
        print("Norm token[0,0] =", float(np.linalg.norm(v_np[0, 0])))

    if not pooled and not tokens:
        print("\nFailed to find 2D/3D tensor — get keys and shape from inspect_elmo.py.")

if __name__ == "__main__":
    main()
