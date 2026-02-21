import tensorflow as tf
import numpy as np

print("TF:", tf.__version__)
print("NumPy:", np.__version__)
print(tf.reduce_sum(np.random.randn(100, 100)))

import tensorflow_hub as hub
hub.load("https://tfhub.dev/google/elmo/3")
print("ELMo OK")