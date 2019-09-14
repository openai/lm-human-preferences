#!/usr/bin/env python3

import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tensorflow.contrib import summary

from lm_human_preferences import label_types, lm_tasks, rewards
from lm_human_preferences.language import trained_models
from lm_human_preferences.policy import Policy
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import gcs, hyperparams
from lm_human_preferences.utils.core import Schema


@dataclass
class LabelHParams(hyperparams.HParams):
    type: str = None
    num_train: int = None
    source: str = None


@dataclass
class RunHParams(hyperparams.HParams):
    seed: Optional[int] = None
    log_interval: int = 10
    save_interval: int = 50
    save_dir: Optional[str] = None

@dataclass
class HParams(hyperparams.HParams):
    run: RunHParams = field(default_factory=RunHParams)

    task: lm_tasks.TaskHParams = field(default_factory=lm_tasks.TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)

    batch_size: int = 40  # total across ranks
    lr: float = 5e-5

    rollout_batch_size: int = 64
    normalize_samples: int = 0  # Samples used to estimate reward mean and std
    debug_normalize: int = 0  # Samples used to check that normalization worked
    # Whether, before training, to normalize the rewards on the policy to the scales on the training buffer.
    # (For comparisons, just use mean 0, var 1.)
    normalize_before: bool = False
    # Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1
    # (so the KL coefficient always has the same meaning).
    normalize_after: bool = False

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        utils.exact_div(self.labels.num_train, self.batch_size)

def round_down_to_multiple(n, divisor):
    return n - n % divisor


def download_labels(source, label_type, question_schemas, total_labels, comm):
    schemas = {**question_schemas, **label_type.label_schemas()}

    """
    if self.is_root:
        with tf.device('cpu:0'):
            self._enqueue_phs = {
                name: tf.placeholder(name=name, dtype=schema.dtype, shape=(None,) + schema.shape)
                for name, schema in self.schemas.items()
            }
            self._enqueue_answers = self.answer_queue.enqueue_many(self._enqueue_phs)
    else:
        self._enqueue_phs = None
        self._enqueue_answers = None
    """

    # TODO: download on just one rank?  then do: labels = utils.mpi_bcast_tensor_dict(labels, comm=comm)
    if source != 'test':
        with open(gcs.download_file_cached(source, comm=comm)) as f:
            results = json.load(f)
            print('Num labels found in source:', len(results))
    else:
        results = [
            {
                name: np.zeros(schema.shape, dtype=schema.dtype.as_numpy_dtype)
                for name, schema in schemas.items()
            }
            for _ in range(50)
        ]

    assert len(results) >= total_labels
    results = results[:total_labels]
    return {k: [a[k] for a in results] for k in schemas.keys()}


class RewardModelTrainer():
    def __init__(self, *, reward_model, policy, query_sampler, hparams, comm):
        self.reward_model = reward_model

        self.policy = policy
        self.hparams = hparams
        self.num_ranks = comm.Get_size()
        self.rank = comm.Get_rank()
        self.comm = comm

        self.label_type = label_types.get(hparams.labels.type)
        self.question_schemas = self.label_type.question_schemas(
            query_length=hparams.task.query_length,
            response_length=hparams.task.response_length,
        )

        data_schemas = {
            **self.question_schemas,
            **self.label_type.label_schemas(),
        }

        with tf.device(None), tf.device('/cpu:0'):
            with tf.variable_scope('label_buffer', use_resource=True, initializer=tf.zeros_initializer):
                self.train_buffer = utils.SampleBuffer(capacity=hparams.labels.num_train, schemas=data_schemas)

        with tf.name_scope('train_reward'):
            summary_writer = utils.get_summary_writer(self.hparams.run.save_dir, subdir='reward_model', comm=comm)

            @utils.graph_function(
                indices=Schema(tf.int32, (None,)),
                lr=Schema(tf.float32, ()))
            def train_batch(indices, lr):
                with tf.name_scope('minibatch'):
                    minibatch = self.train_buffer.read(indices)
                    stats = self.label_type.loss(reward_model=self.reward_model.get_rewards_op, labels=minibatch)

                    train_op = utils.minimize(
                        loss=stats['loss'], lr=lr, params=self.reward_model.get_params(), name='opt', comm=self.comm)

                    with tf.control_dependencies([train_op]):
                        step_var = tf.get_variable(name='train_step', dtype=tf.int64, shape=(), trainable=False, use_resource=True)
                        step = step_var.assign_add(1) - 1

                        stats = utils.FlatStats.from_dict(stats).map_flat(partial(utils.mpi_allreduce_mean, comm=comm)).as_dict()

                        train_stat_op = utils.record_stats(stats=stats, summary_writer=summary_writer, step=step, log_interval=hparams.run.log_interval, comm=comm)

                return train_stat_op
            self.train_batch = train_batch

        if self.hparams.normalize_before or self.hparams.normalize_after:
            @utils.graph_function()
            def target_mean_std():
                """Returns the means and variances to target for each reward model"""
                # Should be the same on all ranks because the train_buf should be the same
                scales = self.label_type.target_scales(self.train_buffer.data())
                if scales is None:
                    return tf.zeros([]), tf.ones([])
                else:
                    mean, var = tf.nn.moments(scales, axes=[0])
                    return mean, tf.sqrt(var)
            self.target_mean_std = target_mean_std

            def stats(query_responses):
                rewards = np.concatenate([self.reward_model.get_rewards(qs, rs) for qs, rs in query_responses], axis=0)
                assert len(rewards.shape) == 1, f'{rewards.shape}'
                sums = np.asarray([rewards.sum(axis=0), np.square(rewards).sum(axis=0)])
                means, sqr_means = self.comm.allreduce(sums, op=MPI.SUM) / (self.num_ranks * rewards.shape[0])
                stds = np.sqrt(sqr_means - means ** 2)
                return means, stds
            self.stats = stats

            def log_stats_after_normalize(stats):
                if comm.Get_rank() != 0:
                    return
                means, stds = stats
                print(f'after normalize: {means} +- {stds}')
            self.log_stats_after_normalize = log_stats_after_normalize

            def reset_reward_scales():
                self.reward_model.reset_reward_scale()
            self.reset_reward_scales = reset_reward_scales

            def set_reward_norms(mean, std, new_mean, new_std):
                print(f'targets: {new_mean} +- {new_std}')
                print(f'before normalize: {mean} +- {std}')
                assert np.isfinite((mean, std, new_mean, new_std)).all()
                self.reward_model.set_reward_norm(old_mean=mean, old_std=std, new_mean=new_mean, new_std=new_std)
            self.set_reward_norms = set_reward_norms

        if self.hparams.normalize_before or self.hparams.normalize_after:
            @utils.graph_function()
            def sample_policy_batch():
                queries = query_sampler('ref_queries')['tokens']
                responses = policy.respond_op(
                    queries=queries, length=hparams.task.response_length)['responses']
                return queries, responses

            def sample_policy_responses(n_samples):
                n_batches = utils.ceil_div(n_samples, hparams.rollout_batch_size)
                return [sample_policy_batch() for _ in range(n_batches)]
            self.sample_policy_responses = sample_policy_responses

        @utils.graph_function(labels=utils.add_batch_dim(data_schemas))
        def add_to_buffer(labels):
            return self.train_buffer.add(**labels)
        self.add_to_buffer = add_to_buffer

    def normalize(self, sample_fn, target_means, target_stds):
        if not self.hparams.normalize_samples:
            return

        self.reset_reward_scales()
        query_responses = sample_fn(self.hparams.normalize_samples)
        means, stds = self.stats(query_responses)

        self.set_reward_norms(means, stds, target_means, target_stds)
        if self.hparams.debug_normalize:
            query_responses = sample_fn(self.hparams.debug_normalize)
            stats = self.stats(query_responses)
            self.log_stats_after_normalize(stats)

    def train(self):
        labels = download_labels(
            self.hparams.labels.source,
            label_type=self.label_type,
            question_schemas=self.question_schemas,
            total_labels=self.hparams.labels.num_train,
            comm=self.comm
        )

        self.add_to_buffer(labels)

        if self.hparams.normalize_before:
            target_mean, target_std = self.target_mean_std()
            self.normalize(self.sample_policy_responses, target_mean, target_std)

        # Collect training data for reward model training.  train_indices will include the indices
        # trained on across all ranks, and its size must be a multiple of minibatch_size.
        per_rank_batch_size = utils.exact_div(self.hparams.batch_size, self.num_ranks)

        # Make sure each rank gets the same shuffle so we train on each point exactly once
        train_indices = self.comm.bcast(np.random.permutation(self.hparams.labels.num_train))

        # Train on train_indices
        print(self.rank, "training on", self.hparams.labels.num_train, "in batches of", per_rank_batch_size)
        for start_index in range(0, self.hparams.labels.num_train, self.hparams.batch_size):
            end_index = start_index + self.hparams.batch_size
            all_ranks_indices = train_indices[start_index:end_index]
            our_indices = all_ranks_indices[self.rank::self.num_ranks]
            lr = (1 - start_index / self.hparams.labels.num_train) * self.hparams.lr
            self.train_batch(our_indices, lr)

        if self.hparams.normalize_after:
            target_mean, target_std = np.zeros([]), np.ones([])
            self.normalize(self.sample_policy_responses, target_mean, target_std)



def train(hparams: HParams):
    with tf.Graph().as_default():
        hyperparams.dump(hparams)
        utils.set_mpi_seed(hparams.run.seed)

        m = trained_models.TrainedModel(hparams.task.policy.initial_model)
        encoder = m.encoding.get_encoder()
        hyperparams.dump(m.hparams(), name='model_hparams')

        comm = MPI.COMM_WORLD
        ref_policy = Policy(
            m, scope='ref_policy',
            is_root=comm.Get_rank() == 0,
            embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
            temperature=hparams.task.policy.temperature,
            build_respond=False)

        reward_model = rewards.RewardModelTrainer(m, is_root=comm.Get_rank() == 0)

        query_sampler = lm_tasks.make_query_sampler(
            hparams=hparams.task, encoder=encoder, comm=comm,
            batch_size=utils.exact_div(hparams.rollout_batch_size, comm.Get_size())
        )

        tf.train.create_global_step()

        reward_trainer = RewardModelTrainer(
            reward_model=reward_model,
            policy=ref_policy,
            query_sampler=query_sampler,
            hparams=hparams,
            comm=comm,
        )

        save_dir = hparams.run.save_dir
        if comm.Get_rank() == 0 and save_dir:
            print(f"Will save to {save_dir}")
            saver = tf.train.Saver(max_to_keep=20, save_relative_paths=True)
            checkpoint_dir = os.path.join(save_dir, 'reward_model/checkpoints/model.ckpt')

            if not save_dir.startswith('gs://'):
                os.makedirs(os.path.join(save_dir, 'reward_model'), exist_ok=True)
            with tf.gfile.Open(os.path.join(save_dir, 'train_reward_hparams.json'), 'w') as f:
                json.dump(hparams.to_nested_dict(), f, indent=2)
            with tf.gfile.Open(os.path.join(save_dir, 'reward_model', 'hparams.json'), 'w') as f:
                json.dump(reward_model.hparams.to_nested_dict(), f, indent=2)
            with tf.gfile.Open(os.path.join(save_dir, 'reward_model', 'encoding'), 'w') as f:
                json.dump(reward_model.trained_model.encoding.name, f, indent=2)
        else:
            saver = None
            checkpoint_dir = None

        with utils.variables_on_gpu():
            init_ops = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer(),
                summary.summary_writer_initializer_op())

            @utils.graph_function()
            def sync_models():
                return utils.variable_synchronizer(comm, vars=ref_policy.get_params() + reward_model.get_params())

        tf.get_default_graph().finalize()

        with utils.mpi_session() as sess:
            init_ops.run()
            sync_models()

            reward_trainer.train()

            if saver:
                saver.save(sess, checkpoint_dir)
