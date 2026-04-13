import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

# two models from TF-Hub:
#    - preprocess (tokenizer + input formatting)
#    - encoder (BERT-Base Uncased)

PREPROCESS_MODEL = "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"
BERT_MODEL      = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-l-12-h-768-a-12/3"

def main():
    # preprocessing + encoder
    preprocess = hub.load(PREPROCESS_MODEL)
    encoder    = hub.load(BERT_MODEL)

    # sentences to embed
    sentences = [
        # "I went to the bank to deposit money.",
        # "The river bank was wide and calm."
        "The central bank raised interest rates yesterday.",
        "Fisherman rested on the river bank at sunset."
    ]

    # preprocessing
    bert_input = preprocess(sentences)

    # BERT
    bert_output = encoder(bert_input)

    # output keys
    print("Output keys:", list(bert_output.keys()))

    # pooled output
    pooled = bert_output["pooled_output"]  # shape (batch, 768)
    print("\nPooled output shape:", pooled.shape)
    print("Pooled vector for first sentence (first 8 dims):", pooled[0][:8].numpy())

    # token-level output
    tokens = bert_output["sequence_output"]  # shape (batch, seq_len, 768)
    print("\nSequence output shape:", tokens.shape)

    # get tokens and their vectors
    for i, sent in enumerate(sentences):
        print(f"\nSentence {i}: '{sent}'")
        # print 5 (size: 768)
        print("Token vectors (first 5 tokens):")
        for j in range(min(5, tokens.shape[1])):
            vec = tokens[i, j].numpy()
            print(f"  token {j} vec[:8]:", vec[:8])

    # cosine similarity
    def cosine(a, b, eps=1e-9):
        a = a / (np.linalg.norm(a) + eps)
        b = b / (np.linalg.norm(b) + eps)
        return float(np.dot(a, b))

    sim = cosine(pooled[0].numpy(), pooled[1].numpy())
    print("\nCosine similarity of pooled sentence embeddings:", sim)


if __name__ == "__main__":
    main()