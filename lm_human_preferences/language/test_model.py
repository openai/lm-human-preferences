#!/usr/bin/env python3
"""Transformer model tests."""

import numpy as np
import tensorflow as tf

from lm_human_preferences.utils import core as utils
from lm_human_preferences.language import model

def test_incremental():
    hparams = model.HParams()
    hparams.override_from_dict(dict(
        n_vocab=10,
        n_ctx=5,
        n_embd=9,
        n_head=3,
        n_layer=2,
    ))
    batch_size = 2
    steps = 5
    np.random.seed(7)
    tf.set_random_seed(7)

    # Transformer model
    m = model.Model(hparams=hparams)
    X = tf.placeholder(shape=[batch_size, None], dtype=tf.int32)
    logits = m(X=X)['lm_logits']
    past_p = tf.placeholder(shape=model.past_shape(hparams=hparams, batch_size=batch_size), dtype=tf.float32)
    # Test reusing it in a different variable scope
    with tf.variable_scope('other_scope'):
        past_lm = m(X=X[:,-1:], past=past_p)
    past_logits = past_lm['lm_logits']
    future = tf.concat([past_p, past_lm['present']], axis=-2)

    # Data
    ids = np.random.randint(hparams.n_vocab, size=[batch_size, steps]).astype(np.int32)
    past = np.zeros(model.past_shape(hparams=hparams, batch_size=batch_size, sequence=0), dtype=np.float32)

    # Evaluate
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for step in range(steps):
            logits_v, past_logits_v, past = sess.run([logits, past_logits, future],
                                                      feed_dict={X: ids[:,:step+1], past_p: past})
            assert np.allclose(logits_v[:,-1:], past_logits_v, atol=1e-3, rtol=1e-3)


def test_mask():
    np.random.seed(7)
    tf.set_random_seed(7)

    # Make a transformer
    hparams = model.HParams()
    hparams.override_from_dict(dict(
        n_vocab=10,
        n_ctx=8,
        n_embd=3,
        n_head=3,
        n_layer=2,
    ))
    batch_size = 4# 64
    policy = model.Model(hparams=hparams)

    # Random pasts and tokens
    past_length = 4
    length = 3
    past = np.random.randn(*model.past_shape(
        hparams=hparams, batch_size=batch_size, sequence=past_length)).astype(np.float32)
    X = np.random.randint(hparams.n_vocab, size=[batch_size, length])

    # Run model without gaps
    logits = policy(past=past, X=X)['lm_logits']

    # Run the same thing, but with gaps randomly inserted
    gap_past_length = 7
    gap_length = 5
    def random_subsequence(*, n, size):
        # Always make the first token be present, since the model tries to fill gaps with the previous states
        sub = [
            np.concatenate(([0], np.random.choice(np.arange(1,n), size=size-1, replace=False)))
            for _ in range(batch_size)
        ]
        return np.sort(sub, axis=-1)
    past_sub = random_subsequence(n=gap_past_length, size=past_length)
    X_sub = random_subsequence(n=gap_length, size=length)
    past_gap = np.random.randn(*model.past_shape(
        hparams=hparams, batch_size=batch_size, sequence=gap_past_length)).astype(np.float32)
    X_gap = np.random.randint(hparams.n_vocab, size=[batch_size, gap_length])
    mask = np.zeros([batch_size, gap_past_length + gap_length], dtype=np.bool)
    for b in range(batch_size):
        for i in range(past_length):
            past_gap[b,:,:,:,past_sub[b,i]] = past[b,:,:,:,i]
        for i in range(length):
            X_gap[b,X_sub[b,i]] = X[b,i]
        mask[b, past_sub[b]] = mask[b, gap_past_length + X_sub[b]] = 1
    gap_logits = policy(past=past_gap, X=X_gap, mask=mask)['lm_logits']
    sub_logits = utils.index_each(gap_logits, X_sub)

    # Compare
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        logits, sub_logits = sess.run([logits, sub_logits])
        assert logits.shape == sub_logits.shape
        assert np.allclose(logits, sub_logits, atol=1e-5)


def test_attention_mask():
    with tf.Session() as sess:
        for nd in 1, 2, 3:
            for ns in range(nd, 4):
                ours = model.attention_mask(nd, ns, dtype=tf.int32)
                theirs = tf.matrix_band_part(tf.ones([nd, ns], dtype=tf.int32), tf.cast(-1, tf.int32), ns-nd)
                ours, theirs = sess.run([ours, theirs])
                print(ours)
                print(theirs)
                assert np.all(ours == theirs)


if __name__ == '__main__':
    test_mask()
    test_attention_mask()
    test_incremental()
