#!/usr/bin/env python3
"""utils tests"""

import numpy as np
import tensorflow as tf

from lm_human_preferences.utils import core as utils


def test_exact_div():
    assert utils.exact_div(12, 4) == 3
    assert utils.exact_div(12, 3) == 4
    try:
        utils.exact_div(7, 3)
        assert False
    except ValueError:
        pass


def test_ceil_div():
    for b in range(1, 10 + 1):
        for a in range(-10, 10 + 1):
            assert utils.ceil_div(a, b) == int(np.ceil(a / b))


def test_expand_tile():
    np.random.seed(7)
    size = 11
    with tf.Session():
        for shape in (), (7,), (3, 5):
            data = np.asarray(np.random.randn(*shape), dtype=np.float32)
            x = tf.constant(data)
            for axis in range(-len(shape) - 1, len(shape) + 1):
                y = utils.expand_tile(x, size, axis=axis).eval()
                assert np.all(np.expand_dims(data, axis=axis) == y)


def test_sample_buffer():
    capacity = 100
    batch = 17
    lots = 100
    with tf.Graph().as_default(), tf.Session() as sess:
        buffer = utils.SampleBuffer(capacity=capacity, schemas=dict(x=utils.Schema(tf.int32, ())))
        tf.variables_initializer(tf.global_variables() + tf.local_variables()).run()
        i_p = tf.placeholder(dtype=tf.int32, shape=())
        add = buffer.add(x=batch * i_p + tf.range(batch))
        sample = buffer.sample(lots, seed=7)['x']
        all_data_1 = buffer.data()
        all_data_2 = buffer.read(tf.range(buffer.size()))
        for i in range(20):
            add.run(feed_dict={i_p: i})
            samples = sample.eval()
            hi = batch * (i + 1)
            lo = max(0, hi - capacity)
            assert lo <= samples.min() <= lo + 3
            assert hi - 5 <= samples.max() < hi
            np.testing.assert_equal(sess.run(all_data_1), sess.run(all_data_2))


def test_where():
    with tf.Session():
        assert np.all(utils.where([False, True], 7, 8).eval() == [8, 7])
        assert np.all(utils.where([False, True, True], [1, 2, 3], 8).eval() == [8, 2, 3])
        assert np.all(utils.where([False, False, True], 8, [1, 2, 3]).eval() == [1, 2, 8])
        assert np.all(utils.where([False, True], [[1, 2], [3, 4]], -1).eval() == [[-1, -1], [3, 4]])
        assert np.all(utils.where([False, True], -1, [[1, 2], [3, 4]]).eval() == [[1, 2], [-1, -1]])


def test_map_flat():
    with tf.Session() as sess:
        inputs = [2], [3, 5], [[7, 11], [13, 17]]
        inputs = map(np.asarray, inputs)
        outputs = sess.run(utils.map_flat(tf.square, inputs))
        for i, o in zip(inputs, outputs):
            assert np.all(i * i == o)


def test_map_flat_bits():
    with tf.Session() as sess:
        inputs = [2], [3, 5], [[7, 11], [13, 17]], [True, False, True]
        dtypes = np.uint8, np.uint16, np.int32, np.int64, np.bool
        inputs = [np.asarray(i, dtype=d) for i, d in zip(inputs, dtypes)]
        outputs = sess.run(utils.map_flat_bits(lambda x: x + 1, inputs))

        def tweak(n):
            return n + sum(2 ** (8 * i) for i in range(n.dtype.itemsize))

        for i, o in zip(inputs, outputs):
            assert np.all(tweak(i) == o)


def test_cumulative_max():
    np.random.seed(7)
    with tf.Session().as_default():
        for x in [
                np.random.randn(10),
                np.random.randn(11, 7),
                np.random.randint(-10, 10, size=10),
                np.random.randint(-10, 10, size=(12, 8)),
                np.random.randint(-10, 10, size=(3, 3, 4)),
        ]:
            assert np.all(utils.cumulative_max(x).eval() == np.maximum.accumulate(x, axis=-1))


def test_index_each():
    np.random.seed(7)
    x = np.random.randn(7, 11)
    i = np.random.randint(x.shape[1], size=x.shape[0])
    y = utils.index_each(x, i)

    x2 = np.random.randn(3, 2, 4)
    i2 = np.random.randint(x2.shape[1], size=x2.shape[0])
    y2 = utils.index_each(x2, i2)

    x3 = np.random.randn(5, 9)
    i3 = np.random.randint(x3.shape[1], size=(x3.shape[0], 2))
    y3 = utils.index_each(x3, i3)
    with tf.Session():
        assert np.all(y.eval() == x[np.arange(7), i])
        assert np.all(y2.eval() == x2[np.arange(3), i2])
        y3val = y3.eval()
        assert np.all(y3val[:,0] == x3[np.arange(5), i3[:,0]])
        assert np.all(y3val[:,1] == x3[np.arange(5), i3[:,1]])


def test_index_each_many():
    np.random.seed(7)
    x = np.random.randn(7, 11)
    i = np.random.randint(x.shape[1], size=[x.shape[0],3])
    y = utils.index_each(x, i)
    with tf.Session():
        assert np.all(y.eval() == x[np.arange(7)[:,None], i])


@utils.graph_function(x=utils.Schema(tf.int32, ()), y=utils.Schema(tf.int32, ()))
def tf_sub(x, y=1):
    return tf.math.subtract(x, y)

@utils.graph_function(x=utils.Schema(tf.int32, ()), y=dict(z1=utils.Schema(tf.int32, ()), z2=utils.Schema(tf.int32, ())))
def tf_sub_2(x, y):
    return tf.math.subtract(x, y['z1']) - y['z2']

def test_graph_function():
    with tf.Session().as_default():
        assert tf_sub(3) == 2
        assert tf_sub(x=3) == 2
        assert tf_sub(5, 2) == 3
        assert tf_sub(y=2, x=5) == 3
        assert tf_sub_2(5, dict(z1=1, z2=2)) == 2

def test_top_k():
    with tf.Session().as_default():
        logits = tf.constant([[[1,1.01,1.001,0,0,0,2]]], dtype=tf.float32)
        np.testing.assert_allclose(
            utils.take_top_k_logits(logits, 1).eval(),
            [[[-1e10,-1e10,-1e10,-1e10,-1e10,-1e10,2]]]
        )
        np.testing.assert_allclose(
            utils.take_top_k_logits(logits, 2).eval(),
            [[[-1e10,1.01,-1e10,-1e10,-1e10,-1e10,2]]]
        )
        np.testing.assert_allclose(
            utils.take_top_k_logits(logits, 3).eval(),
            [[[-1e10,1.01,1.001,-1e10,-1e10,-1e10,2]]]
        )
        np.testing.assert_allclose(
            utils.take_top_k_logits(logits, 4).eval(),
            [[[1,1.01,1.001,-1e10,-1e10,-1e10,2]]]
        )
        np.testing.assert_allclose(
            utils.take_top_k_logits(logits, 5).eval(),
            [[[1,1.01,1.001,0,0,0,2]]]
        )


def test_top_p():
    with tf.Session().as_default():
        logits = tf.constant([[[1,1.01,1.001,0,0,0,2]]], dtype=tf.float32)
        np.testing.assert_allclose(
            utils.take_top_p_logits(logits, 1).eval(),
            logits.eval()
        )
        np.testing.assert_allclose(
            utils.take_top_p_logits(logits, 0).eval(),
            [[[-1e10,-1e10,-1e10,-1e10,-1e10,-1e10,2]]]
        )
        np.testing.assert_allclose(
            utils.take_top_p_logits(logits, 0.7).eval(),
            [[[-1e10,1.01,1.001,-1e10,-1e10,-1e10,2]]]
        )
        np.testing.assert_allclose(
            utils.take_top_p_logits(logits, 0.6).eval(),
            [[[-1e10,1.01,-1e10,-1e10,-1e10,-1e10,2]]]
        )
        np.testing.assert_allclose(
            utils.take_top_p_logits(logits, 0.5).eval(),
            [[[-1e10,-1e10,-1e10,-1e10,-1e10,-1e10,2]]]
        )

def test_safe_zip():
    assert list(utils.safe_zip([1, 2], [3, 4])) == [(1, 3), (2, 4)]
    try:
        utils.safe_zip([1, 2], [3, 4, 5])
        assert False
    except ValueError:
        pass


if __name__ == '__main__':
    test_sample_buffer()
    test_cumulative_max()
    test_where()
    test_index_each()
    test_graph_function()
    test_top_k()
    test_top_p()
    test_safe_zip()
