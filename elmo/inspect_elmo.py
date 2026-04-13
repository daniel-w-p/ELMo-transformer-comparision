import tensorflow as tf
import tensorflow_hub as hub

HANDLE = "https://www.kaggle.com/models/google/elmo/TensorFlow1/elmo/3"

def main():
    elmo = hub.load(HANDLE)

    # Print "signatures" from SavedModel?
    sigs = list(elmo.signatures.keys())
    print("Available signatures:", sigs)

    # default if exits
    sig_name = "default" if "default" in elmo.signatures else sigs[0]
    f = elmo.signatures[sig_name]
    print("Using signature:", sig_name)

    # batch strings # TODO make more
    x = tf.constant([
        "I love pizza",
        "I love pasta"
    ])

    y = f(x)

    # Outputs
    print("Output keys:", list(y.keys()))
    for k, v in y.items():
        print(f"- {k:>15s} | dtype={v.dtype} | shape={v.shape}")



if __name__ == "__main__":
    main()