"""Synthetic scores."""

import os

import tensorflow as tf
from mpi4py import MPI

from lm_human_preferences.language import trained_models, model
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils.core import Schema


# TODO: combine this with TrainedRewardModel
class RewardModelTrainer:
    def __init__(
            self,
            trained_model, *,
            scope='reward_model', use_resource=False,
            is_root=True,
    ):
        self.trained_model = trained_model
        self.hparams = trained_model.hparams()
        self.is_root = is_root

        self.use_resource = use_resource
        self.encoder = self.trained_model.encoding.get_encoder()

        self.scope = scope
        self.model = model.Model(hparams=self.hparams, scope=f'{scope}/model', scalar_heads=['reward'])

        self.built = False
        self.padding_token = self.encoder.padding_token

        self.get_rewards = utils.graph_function(
            queries=Schema(tf.int32, (None, None)),
            responses=Schema(tf.int32, (None, None)),
        )(self.get_rewards_op)


    def get_encoder(self):
        return self.encoder

    def _build(self, tokens, do_dropout=False, name=None):
        with tf.variable_scope(self.scope, reuse=self.built, auxiliary_name_scope=not self.built, use_resource=self.use_resource):
            lm_output = self.model(X=tokens, do_dropout=do_dropout, padding_token=self.padding_token)

            reward = lm_output['reward'][:, -1]
            with tf.variable_scope('reward_norm'):
                if not self.built:
                    self.reward_gain = tf.get_variable('gain', shape=(), initializer=tf.constant_initializer(1))
                    self.reward_bias = tf.get_variable('bias', shape=(), initializer=tf.constant_initializer(0))
                    self._reward_gain_p = tf.placeholder(name='gain_p', dtype=tf.float32, shape=())
                    self._reward_bias_p = tf.placeholder(name='bias_p', dtype=tf.float32, shape=())
                    self._set_reward_norm = tf.group(self.reward_gain.assign(self._reward_gain_p),
                                                     self.reward_bias.assign(self._reward_bias_p))
                if reward is not None:
                    reward = self.reward_gain * reward + self.reward_bias
            if not self.built:
                self._set_initializers()
            self.built = True
            return reward

    def ensure_built(self):
        if self.built:
            return
        with tf.name_scope('dummy'):
            self._build(tokens=tf.zeros([0,0], dtype=tf.int32))

    def get_params(self):
        self.ensure_built()
        return self.model.get_params() + [self.reward_gain, self.reward_bias]

    def reset_reward_scale(self):
        sess = tf.get_default_session()
        sess.run(self._set_reward_norm, feed_dict={self._reward_gain_p: 1, self._reward_bias_p: 0})

    def set_reward_norm(self, *, old_mean, old_std, new_mean, new_std):
        """Given old_mean+-old_std of reward_model, change gain and bias to get N(new_mean,new_std)."""
        sess = tf.get_default_session()
        old_gain, old_bias = sess.run((self.reward_gain, self.reward_bias))
        assert old_gain == 1 and old_bias == 0,\
            f'set_reward_norm expects gain = 1 and bias = 0, not {old_gain}, {old_bias}'
        # gain * N(old_mean,old_std) + bias = N(gain * old_mean, gain * old_std) + bias
        #                                   = N(gain * old_mean + bias, gain * old_std)
        # gain * old_std = new_std, gain = new_std / old_std
        # gain * old_mean + bias = new_mean, bias = new_mean - gain * old_mean
        gain = new_std / old_std
        bias = new_mean - gain * old_mean
        sess.run(self._set_reward_norm, feed_dict={self._reward_gain_p: gain, self._reward_bias_p: bias})

    def _set_initializers(self):
        """Change initializers to load a language model from a tensorflow checkpoint."""
        # Skip if
        # 1. We're not rank 0.  Values will be copied from there.
        # 2. We want random initialization.  Normal initialization will do the work.
        if not self.is_root or self.trained_model.name == 'test':
            return

        with tf.init_scope():
            # Initialize!
            params = {v.op.name: v for v in utils.find_trainable_variables(self.scope)}
            assert params
            self.trained_model.init_op(params, new_scope=self.scope)

    def get_rewards_op(self, queries, responses):
        tokens = tf.concat([queries, responses], axis=1)
        return self._build(tokens)


class TrainedRewardModel():
    def __init__(self, train_dir, encoding, *, scope='reward_model', comm=MPI.COMM_WORLD):
        self.train_dir = train_dir
        self.comm = comm

        self.encoding = encoding
        encoder = encoding.get_encoder()
        if train_dir != 'test':
            self.hparams = trained_models.load_hparams(os.path.join(train_dir, 'hparams.json'))
            assert self.hparams.n_vocab == encoding.n_vocab, f'{self.hparams.n_vocab} != {encoding.n_vocab}'
        else:
            self.hparams = trained_models.test_hparams()

        self.padding_token = encoder.padding_token

        self.encoder = encoder

        self.scope = scope
        self.model = model.Model(hparams=self.hparams, scope=f'{scope}/model', scalar_heads=['reward'])

    def _build(self, X):
        results = self.model(X=X, padding_token=self.padding_token)
        reward = results['reward'][:, -1]
        with tf.variable_scope(f'{self.scope}/reward_norm'):
            self.reward_gain = tf.get_variable('gain', shape=(), initializer=tf.constant_initializer(1))
            self.reward_bias = tf.get_variable('bias', shape=(), initializer=tf.constant_initializer(0))
        reward = self.reward_gain * reward + self.reward_bias
        self._set_initializers()
        return reward

    def ensure_built(self):
        if self.model.built:
            return
        with tf.name_scope('dummy'):
            self._build(X=tf.zeros([0,0], dtype=tf.int32))

    def _set_initializers(self):
        """Change initializers to load a model from a tensorflow checkpoint."""
        if self.comm.Get_rank() > 0 or self.train_dir == 'test':
            return

        assert self.model.built
        checkpoint_scope = 'reward_model'

        with tf.init_scope():
            # Initialize!
            params = {v.op.name: v for v in self.get_params()}
            checkpoint = tf.train.latest_checkpoint(os.path.join(self.train_dir, 'checkpoints/'))
            available = tf.train.list_variables(checkpoint)
            unchanged = {}

            for name, shape in available:
                if not name.startswith(checkpoint_scope + '/'):
                    # print('skipping', name)
                    continue
                if name.endswith('adam') or name.endswith('adam_1'):
                    # print('skipping', name)
                    continue
                print('setting', name)
                var = params[self.scope + name[len(checkpoint_scope):]]
                assert var.shape == shape, 'Shape mismatch: %s.shape = %s != %s' % (var.op.name, var.shape, shape)
                unchanged[name] = var
            tf.train.init_from_checkpoint(checkpoint, unchanged)

    def get_params(self):
        return self.model.get_params() + [self.reward_gain, self.reward_bias]

    def score_fn(self, queries, responses):
        tokens = tf.concat([queries, responses], axis=1)
        return self._build(tokens)
