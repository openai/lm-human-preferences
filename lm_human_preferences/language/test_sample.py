#!/usr/bin/env python3
"""Test sample_sequence()."""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

from lm_human_preferences.language import sample

n_vocab = 10
batch_size = 2
hparams = HParams(
    n_layer=0,
    n_head=1,
    n_embd=0,
    n_attn=0,
)

# Returns a policy that deterministically chooses previous token + 1.
def step(hparams, tokens, past=None, past_tokens=None):
    logits = tf.one_hot(tokens + 1, n_vocab, on_value=0., off_value=-np.inf, dtype=tf.float32)
    ret = {
        'logits': logits,
        'presents': tf.zeros(shape=[2, 0, 2, 1, 0, 0]),
        }
    return ret

def test_sample_sequence():
    output = sample.sample_sequence(step=step, model_hparams=hparams, length=4, batch_size=batch_size,
                                    context=tf.constant([[5, 0], [4, 3]]))
    expected = np.array([[5, 0, 1, 2, 3, 4], [4, 3, 4, 5, 6, 7]])

    with tf.Session() as sess:
        np.testing.assert_array_equal(sess.run(output)['tokens'], expected)


if __name__ == '__main__':
    test_sample_sequence()
