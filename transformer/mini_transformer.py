import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        positions = self.pos_emb(positions)
        return self.token_emb(x) + positions


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x, mask=None, return_attention=False):
        # Self-attention
        attn_scores = None
        attn_output = self.att(x, x, attention_mask=mask, return_attention_scores=return_attention)

        if return_attention:
            attn_out, attn_scores = attn_output
        else:
            attn_out = attn_output

        x1 = self.ln1(x + attn_out)

        # Feed Forward
        ffn_output = self.ffn(x1)
        x2 = self.ln2(x1 + ffn_output)

        if return_attention:
            return x2, attn_scores
        return x2


class MiniTransformer(tf.keras.Model):
    def __init__(self, vocab_size, max_len, embed_dim=64, num_heads=4, ff_dim=128):
        super().__init__()
        self.embedding = PositionalEmbedding(max_len, vocab_size, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)

    def call(self, x, mask=None, return_attention=False):
        x = self.embedding(x)
        if return_attention:
            return self.transformer(x, mask=mask, return_attention=True)
        return self.transformer(x, mask=mask)