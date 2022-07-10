# Official tensorflow word2vec tutorial
# https://www.tensorflow.org/tutorials/text/word2vec

from data import get_training_data
from tensorflow.keras import layers
from ipdb import set_trace

import tensorflow as tf

num_ns = 4
skipgram_window_size = 3
vocab_size = 4096
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
embedding_dim = 128
_targets, _contexts, _labels = get_training_data(
    skipgram_window_size=skipgram_window_size
)


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=1, name="w2v_embedding"
        )
        self.context_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=num_ns + 1
        )

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)

        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        # print("target: ", target, "context: ", context)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum("be,bce->bc", word_emb, context_emb)
        print("dots.shape", word_emb.shape, context_emb.shape, dots.shape)
        # dots: (batch, context)
        return dots


def ce_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


dataset = tf.data.Dataset.from_tensor_slices(
    ((_targets[:-32], _contexts[:-32]), _labels[:-32])
)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
print(dataset)

word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])


test_targets = tf.constant(_targets[:BATCH_SIZE])
test_contexts = tf.constant(_contexts[:BATCH_SIZE])
out = word2vec.predict((test_targets, test_contexts))
