import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from ipdb import set_trace


def get_training_data(skipgram_window_size=2):
    # Load the TensorBoard notebook extension
    # %load_ext tensorboard
    SEED = 42
    AUTOTUNE = tf.data.AUTOTUNE

    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
    print(sampling_table)

    # Generates skip-gram pairs with negative sampling for a list of sequences
    # (int-encoded sentences) based on window size, number of negative samples
    # and vocabulary size.
    def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []

        # Build the sampling table for `vocab_size` tokens.
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

        # Iterate over all sequences (sentences) in the dataset.
        for sequence in tqdm.tqdm(sequences):
            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0,
            )

            # Iterate over each positive skip-gram pair to produce training examples
            # with a positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype="int64"), 1
                )
                (
                    negative_sampling_candidates,
                    _,
                    _,
                ) = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=vocab_size,
                    seed=SEED,
                    name="negative_sampling",
                )

                # Build context and label vectors (for one target word)
                negative_sampling_candidates = tf.expand_dims(
                    negative_sampling_candidates, 1
                )

                context = tf.concat([context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0] * num_ns, dtype="int64")

                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        return targets, contexts, labels

    path_to_file = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
    )

    text_ds = tf.data.TextLineDataset(path_to_file).filter(
        lambda x: tf.cast(tf.strings.length(x), bool)
    )

    # Now, create a custom standardization function to lowercase the text and
    # remove punctuation.
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(
            lowercase, "[%s]" % re.escape(string.punctuation), ""
        )

    # Define the vocabulary size and the number of words in a sequence.
    vocab_size = 4096
    sequence_length = 10

    # Use the `TextVectorization` layer to normalize, split, and map strings to
    # integers. Set the `output_sequence_length` length to pad all samples to the
    # same length.
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    vectorize_layer.adapt(text_ds.batch(1024))

    # Save the created vocabulary for reference.enecccbnlvkrkrugecjibhnhjjnnkrncfevfnlgcblhe

    inverse_vocab = vectorize_layer.get_vocabulary()
    print(inverse_vocab[:20])

    # Vectorize the data in text_ds.
    text_vector_ds = (
        text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
    )
    sequences = list(text_vector_ds.as_numpy_iterator())

    _targets, _contexts, _labels = generate_training_data(
        sequences=sequences,
        window_size=skipgram_window_size,
        num_ns=4,
        vocab_size=vocab_size,
        seed=SEED,
    )

    _targets = np.array(_targets)
    _contexts = np.array(_contexts)[:, :, 0]
    _labels = np.array(_labels)

    print("\n")
    print(f"targets.shape: {_targets.shape}")
    print(f"contexts.shape: {_contexts.shape}")
    print(f"labels.shape: {_labels.shape}")

    return _targets, _contexts, _labels
