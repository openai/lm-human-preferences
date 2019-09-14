"""Alec's transformer model."""

from functools import partial
from typing import Optional
from dataclasses import dataclass

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import function

from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import hyperparams

@dataclass
class HParams(hyperparams.HParams):
    # Encoding (set during loading process)
    n_vocab: int = 0

    # Model parameters
    n_ctx: int = 512
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    head_pdrop: float = 0.1


def parse_comma_separated_int_list(s):
    return [int(i) for i in s.split(",")] if s else []


def gelu(x):
    with tf.name_scope('gelu'):
        return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def dropout(x, pdrop, *, do_dropout, stateless=True, seed=None, name):
    """Like tf.nn.dropout but stateless.
    """
    if stateless:
        assert seed is not None
    def _dropout():
        with tf.name_scope(name):
            noise_shape = tf.shape(x)

            if stateless:
                r = tf.random.stateless_uniform(noise_shape, seed, dtype=x.dtype)
                # floor uniform [keep_prob, 1.0 + keep_prob)
                mask = tf.floor(1 - pdrop + r)
                return x * (mask * (1 / (1 - pdrop)))
            else:
                return tf.nn.dropout(x, rate=pdrop, noise_shape=noise_shape)
    if pdrop == 0 or not do_dropout:
        return x
    else:
        return _dropout()


def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        s = tf.reduce_mean(tf.square(x), axis=axis, keepdims=True)
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = s - tf.square(u)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = utils.shape_list(x)
    return tf.reshape(x, start + [n, m//n])


def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = utils.shape_list(x)
    return tf.reshape(x, start + [a*b])


def conv1x1(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = utils.shape_list(x)

        # Don't cast params until just prior to use -- saves a lot of memory for large models
        with tf.control_dependencies([x]):
            w = tf.squeeze(tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev)), axis=0)
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.matmul(tf.reshape(x, [-1, nx]), w) + b
        c = tf.reshape(c, start+[nf])
        return c


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    # to ignore first parts of context (useful for sampling with static shapes)
    # m = tf.math.logical_and(m, tf.math.logical_or(j  >= ignore, i < ignore - ns + nd))
    return tf.cast(m, dtype)


def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def attn(x, scope, n_state, *, past, mask, do_dropout, scale=False, hparams, seed):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        bs, _, nd, ns = utils.shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        if mask is not None:
            b *= tf.reshape(tf.cast(mask, w.dtype), [bs, 1, 1, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v, *, seed):
        orig_dtype = v.dtype
        q, k, v = map(partial(tf.cast, dtype=tf.float32), (q, k, v))
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)

        if scale:
            n_state = v.shape[-1].value
            w = w * tf.rsqrt(tf.cast(n_state, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, hparams.attn_pdrop,
                    do_dropout=do_dropout, name='attn_drop', stateless=True, seed=seed)
        a = tf.matmul(w, v)
        a = tf.cast(a, dtype=orig_dtype, name='a_cast')
        return a

    with tf.variable_scope(scope):
        attn_seed, resid_seed = split_seed(seed, 2)

        assert n_state % hparams.n_head == 0
        w_init_stdev = 1/np.sqrt(n_state)
        c = conv1x1(x, 'c_attn', n_state * 3, w_init_stdev=w_init_stdev)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v, seed=attn_seed)
        a = merge_heads(a)
        w_init_stdev = 1/np.sqrt(n_state*hparams.n_layer)
        a = conv1x1(a, 'c_proj', n_state, w_init_stdev=w_init_stdev)
        a = dropout(a, hparams.resid_pdrop, do_dropout=do_dropout, stateless=True, seed=resid_seed, name='attn_resid_drop')
        return a, present


def mlp(x, scope, n_hidden, *, do_dropout, hparams, seed):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        w_init_stdev = 1/np.sqrt(nx)
        h = gelu(
            conv1x1(x, 'c_fc', n_hidden, w_init_stdev=w_init_stdev))
        w_init_stdev = 1/np.sqrt(n_hidden*hparams.n_layer)
        h2 = conv1x1(h, 'c_proj', nx, w_init_stdev=w_init_stdev)
        h2 = dropout(h2, hparams.resid_pdrop, do_dropout=do_dropout, stateless=True, seed=seed, name='mlp_drop')
        return h2


def block(x, scope, *, past, mask, do_dropout, scale=False, hparams, seed):
    with tf.variable_scope(scope):
        attn_seed, mlp_seed = split_seed(seed, 2)

        nx = x.shape[-1].value
        a, present = attn(
            norm(x, 'ln_1'),
            'attn', nx, past=past, mask=mask, do_dropout=do_dropout, scale=scale, hparams=hparams, seed=attn_seed)
        x = x + a

        m = mlp(
            norm(x, 'ln_2'),
            'mlp', nx*4, do_dropout=do_dropout, hparams=hparams, seed=mlp_seed)
        h = x + m
        return h, present


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """Force gradient to be a dense tensor.

    It's often faster to do dense embedding gradient on GPU than sparse on CPU.
    """
    return x


def embed(X, we):
    """Embedding lookup.

    X has shape [batch, sequence, info].  Currently info = 2 corresponding to [token_id, position].
    """
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    return e


#tensor contraction of the final axes of x with the first axes of y
#need to write it ourselves because tensorflow's tensordot is slow
def tensordot(x, y, num_axes):
    split_x_axes_at = x.shape.ndims - num_axes
    x_shape = tf.shape(x)[:split_x_axes_at]
    y_shape = tf.shape(y)[num_axes:]
    rx = tf.reshape(x, [tf.reduce_prod(x_shape), tf.reduce_prod(tf.shape(x)[split_x_axes_at:])])
    ry = tf.reshape(y, [-1, tf.reduce_prod(y_shape)])
    rresult = tf.matmul(rx, ry)
    result = tf.reshape(rresult, tf.concat([x_shape, y_shape], axis=0))
    result.set_shape(x.shape[:split_x_axes_at].concatenate(y.shape[num_axes:]))
    return result


#more convenient fc layer that avoids stupid shape stuff
#consumes in_axes of x
#produces y of shape outshape
def fc_layer(x, outshape, *, in_axes=1, scale=None):
    inshape = tuple([int(d) for d in x.shape[-in_axes:]]) if in_axes>0 else ()
    outshape = tuple(outshape)
    if scale is None:
        scale = 1 / np.sqrt(np.prod(inshape) + 1)
    w = tf.get_variable('w', inshape + outshape, initializer=tf.random_normal_initializer(stddev=scale))
    b = tf.get_variable('b', outshape, initializer=tf.constant_initializer(0))
    # Call the regularizer manually so that it works correctly with GradientTape
    regularizer = tf.contrib.layers.l2_regularizer(scale=1/np.prod(outshape)) #so that initial value of regularizer is 1
    reg_loss = regularizer(w)
    return tensordot(x, w, in_axes) + b, reg_loss


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, utils.exact_div(hparams.n_embd, hparams.n_head)]


def positions_for(*, batch, sequence, past_length, mask):
    if mask is None:
        return utils.expand_tile(past_length + tf.range(sequence), batch, axis=0)
    else:
        return tf.cumsum(tf.cast(mask, tf.int32), exclusive=True, axis=-1)[:, past_length:]


def split_seed(seed, n=2):
    if n == 0:
        return []
    return tf.split(
        tf.random.stateless_uniform(dtype=tf.int64, shape=[2*n], minval=-2**63, maxval=2**63-1, seed=seed),
        n, name='split_seeds')


class Model:
    def __init__(self, hparams: HParams, scalar_heads=[], scope=None):
        self.hparams = hparams
        self.scalar_heads = scalar_heads
        with tf.variable_scope(scope, 'model') as scope:
            self.scope = scope
        self.built = False

    def __call__(self, *, X, Y=None, past=None, past_tokens=None, mask=None,
                 padding_token: Optional[int]=None, do_dropout=False):
        X = tf.convert_to_tensor(X, dtype=tf.int32)
        if mask is not None:
            mask = tf.convert_to_tensor(mask, dtype=tf.bool)
            assert mask.dtype == tf.bool
        if padding_token is not None:
            assert mask is None, 'At most one of mask and padding_token should be set'
            mask = tf.not_equal(X, padding_token)
            X = tf.where(mask, X, tf.zeros_like(X))
            if past is not None:
                assert past_tokens is not None, 'padding_token requires past_tokens'
                mask = tf.concat([tf.not_equal(past_tokens, padding_token), mask], axis=1)
        with tf.variable_scope(self.scope, reuse=self.built, auxiliary_name_scope=not self.built):
            self.built = True
            results = {}
            batch, sequence = utils.shape_list(X)

            seed = tf.random.uniform(dtype=tf.int64, shape=[2], minval=-2**63, maxval=2**63-1)
            wpe_seed, wte_seed, blocks_seed, heads_seed = split_seed(seed, 4)

            wpe = tf.get_variable('wpe', [self.hparams.n_ctx, self.hparams.n_embd],
                                  initializer=tf.random_normal_initializer(stddev=0.01))
            wte = tf.get_variable('wte', [self.hparams.n_vocab, self.hparams.n_embd],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
            wpe = dropout(wpe, self.hparams.embd_pdrop,
                          do_dropout=do_dropout, stateless=True, seed=wpe_seed, name='wpe_drop')
            wte = dropout(wte, self.hparams.embd_pdrop,
                          do_dropout=do_dropout, stateless=True, seed=wte_seed, name='wte_drop')

            past_length = 0 if past is None else tf.shape(past)[-2]

            positions = positions_for(batch=batch, sequence=sequence, past_length=past_length, mask=mask)
            h = embed(X, wte) + embed(positions, wpe)
            # Transformer
            presents = []
            pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
            assert len(pasts) == self.hparams.n_layer
            block_seeds = split_seed(blocks_seed, self.hparams.n_layer)
            for layer, (past, block_seed) in enumerate(zip(pasts, block_seeds)):
                h, present = block(
                    h, 'h%d' % layer, past=past, mask=mask, do_dropout=do_dropout, scale=True,
                    hparams=self.hparams, seed=block_seed)
                presents.append(present)
            results['present'] = tf.stack(presents, axis=1)
            h = norm(h, 'ln_f')
            if mask is not None:
                # For non-present tokens, use the output from the last present token instead.
                present_indices = utils.where(mask[:,past_length:], tf.tile(tf.range(sequence)[None,:], [batch, 1]), -1)
                use_indices = utils.cumulative_max(present_indices)
                # assert since GPUs don't
                with tf.control_dependencies([tf.assert_none_equal(use_indices, -1)]):
                    h = utils.index_each(h, use_indices)
            results['h'] = h

            # Language model loss.  Do tokens <n predict token n?
            h_flat = tf.reshape(h, [batch*sequence, self.hparams.n_embd])
            flat_lm_logits = tf.matmul(h_flat, wte, transpose_b=True)

            labels = tf.concat([X[:, 1:], X[:, :1]], axis=1)
            flat_labels = tf.reshape(labels, [batch*sequence])

            flat_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=flat_labels,
                logits=flat_lm_logits)

            lm_losses = tf.reshape(flat_losses, [batch, sequence])
            lm_logits = tf.reshape(flat_lm_logits, [batch, sequence, -1])

            relevant_losses = lm_losses[:, :-1]
            results['lm_all_losses'] = relevant_losses
            results['lm_logits'] = lm_logits
            results['lm_losses'] = tf.reduce_mean(relevant_losses, axis=-1)

            head_seeds = split_seed(heads_seed, len(self.scalar_heads))
            for head_name, head_seed in zip(self.scalar_heads, head_seeds):
                with tf.variable_scope(f"heads/{head_name}"):
                    dropped_h = \
                        dropout(h, self.hparams.head_pdrop, do_dropout=do_dropout, seed=head_seed, name='drop')
                    # TODO: refactor this, perhaps move to Policy
                    res, reg_loss = fc_layer(dropped_h, (), scale=0 if head_name == 'value' else None)
                    results[head_name] = tf.cast(res, dtype=tf.float32, name='res_cast')
                    results[f"{head_name}_regularizer"] = tf.cast(reg_loss, dtype=tf.float32, name='reg_loss_cast')

            # All done!
            return results

    def get_params(self):
        assert self.built
        params = utils.find_trainable_variables(self.scope.name)
        assert len(params) > 0
        return params
