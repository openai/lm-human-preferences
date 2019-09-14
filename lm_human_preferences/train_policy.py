#!/usr/bin/env python3

import json
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tensorflow.contrib import summary

from lm_human_preferences import lm_tasks, train_reward
from lm_human_preferences.language import trained_models
from lm_human_preferences.policy import Policy
from lm_human_preferences.rewards import TrainedRewardModel
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import hyperparams
from lm_human_preferences.utils.core import Schema


@dataclass
class AdaptiveKLParams(hyperparams.HParams):
    target: float = None
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams(hyperparams.HParams):
    kl_coef: float = 0.2
    adaptive_kl: Optional[AdaptiveKLParams] = None

    trained_model: Optional[str] = None

    train_new_model: Optional[train_reward.HParams] = None

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        assert self.trained_model is None or self.train_new_model is None, 'Cannot use trained_model and train new model'
        assert self.trained_model is not None or self.train_new_model is not None, 'Need either trained_model or to train a new model'


@dataclass
class PpoHParams(hyperparams.HParams):
    total_episodes: int = 2000000
    batch_size: int = 64
    nminibatches: int = 1
    noptepochs: int = 4
    lr: float = 5e-6
    vf_coef: float = .1
    cliprange: float = .2
    cliprange_value: float = .2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = True


@dataclass
class HParams(hyperparams.HParams):
    run: train_reward.RunHParams = field(default_factory=train_reward.RunHParams)

    task: lm_tasks.TaskHParams = field(default_factory=lm_tasks.TaskHParams)
    rewards: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        # NOTE: must additionally divide by # ranks
        minibatch_size = utils.exact_div(self.ppo.batch_size, self.ppo.nminibatches)
        if self.ppo.whiten_rewards:
            assert minibatch_size >= 8, \
                f"Minibatch size {minibatch_size} is insufficient for whitening in PPOTrainer.loss"


def nupdates(hparams):
    return utils.ceil_div(hparams.ppo.total_episodes, hparams.ppo.batch_size)


def policy_frac(hparams):
    """How far we are through policy training."""
    return tf.cast(tf.train.get_global_step(), tf.float32) / nupdates(hparams)


def tf_times():
    """Returns (time since start, time since last) as a tensorflow op."""
    # Keep track of start and last times
    with tf.init_scope():
        init = tf.timestamp()

    def make(name):
        return tf.Variable(init, name=name, trainable=False, use_resource=True)

    start = make('start_time')
    last = make('last_time')

    # Get new time and update last
    now = tf.timestamp()
    prev = last.read_value()
    with tf.control_dependencies([prev]):
        with tf.control_dependencies([last.assign(now)]):
            return tf.cast(now - start.read_value(), tf.float32), tf.cast(now - prev, tf.float32)


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class AdaptiveKLController:
    def __init__(self, init_kl_coef, hparams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult



class PPOTrainer():
    def __init__(self, *, policy, ref_policy, query_sampler, score_fn, hparams, comm):
        self.comm = comm
        self.policy = policy
        self.ref_policy = ref_policy
        self.score_fn = score_fn
        self.hparams = hparams

        if hparams.rewards.adaptive_kl is None:
            self.kl_ctl = FixedKLController(hparams.rewards.kl_coef)
        else:
            self.kl_ctl = AdaptiveKLController(hparams.rewards.kl_coef, hparams=hparams.rewards.adaptive_kl)

        response_length = hparams.task.response_length
        query_length = hparams.task.query_length

        @utils.graph_function()
        def sample_queries():
            return query_sampler()['tokens']
        self.sample_queries = sample_queries

        def compute_rewards(scores, logprobs, ref_logprobs):
            kl = logprobs - ref_logprobs
            non_score_reward = -self.kl_ctl.value * kl
            rewards = non_score_reward.copy()
            rewards[:, -1] += scores
            return rewards, non_score_reward, self.kl_ctl.value
        self.compute_rewards = compute_rewards

        # per rank sizes
        per_rank_rollout_batch_size = utils.exact_div(hparams.ppo.batch_size, comm.Get_size())
        per_rank_minibatch_size = utils.exact_div(per_rank_rollout_batch_size, hparams.ppo.nminibatches)

        @utils.graph_function(
            rollouts=dict(
                queries=Schema(tf.int32, (per_rank_minibatch_size, query_length)),
                responses=Schema(tf.int32, (per_rank_minibatch_size, response_length)),
                values=Schema(tf.float32, (per_rank_minibatch_size, response_length)),
                logprobs=Schema(tf.float32, (per_rank_minibatch_size, response_length)),
                rewards=Schema(tf.float32, (per_rank_minibatch_size, response_length)),
            ))
        def train_minibatch(rollouts):
            """One step of PPO training."""

            left = 1 - policy_frac(hparams)
            lrnow = hparams.ppo.lr * left

            ppo_loss, stats = self.loss(rollouts)
            ppo_train_op = utils.minimize(
                loss=ppo_loss, lr=lrnow, params=policy.get_params(), name='ppo_opt', comm=self.comm)
            return ppo_train_op, stats

        def train(rollouts):
            stat_list = []

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(hparams.ppo.noptepochs):
                order = np.random.permutation(per_rank_rollout_batch_size)
                for mb_start in range(0, per_rank_rollout_batch_size, per_rank_minibatch_size):
                    mb_data = {k: v[order[mb_start:mb_start+per_rank_minibatch_size]]
                               for k, v in rollouts.items()}

                    step = tf.train.get_global_step().eval()

                    _, stats = train_minibatch(mb_data)
                    stat_list.append(stats)

            # Collect the stats. (They will be averaged later.)
            return {k: [s[k] for s in stat_list] for k in stat_list[0].keys()}
        self.train = train

        # NOTE: must line up with stats created in self.loss (TODO: better solution?)
        scalar_batch = Schema(tf.float32, (None,))
        ppo_stat_schemas = utils.flatten_dict(dict(
            loss=dict(policy=scalar_batch, value=scalar_batch, total=scalar_batch),
            policy=dict(entropy=scalar_batch, approxkl=scalar_batch, clipfrac=scalar_batch),
            returns=dict(mean=scalar_batch, var=scalar_batch),
            val=dict(vpred=scalar_batch, error=scalar_batch, clipfrac=scalar_batch, mean=scalar_batch, var=scalar_batch),
        ), sep='/')
        stat_data_schemas = dict(
            logprobs=Schema(tf.float32, (None, hparams.task.response_length)),
            ref_logprobs=Schema(tf.float32, (None, hparams.task.response_length)),
            scores=scalar_batch,
            non_score_reward=Schema(tf.float32, (None, hparams.task.response_length)),
            score_stats=score_fn.stat_schemas,
            train_stats=ppo_stat_schemas,
        )
        @utils.graph_function(
            **stat_data_schemas, kl_coef=Schema(tf.float32, ()))
        def record_step_stats(*, kl_coef, **data):
            ppo_summary_writer = utils.get_summary_writer(self.hparams.run.save_dir, subdir='ppo', comm=self.comm)

            kl = data['logprobs'] - data['ref_logprobs']
            mean_kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
            mean_entropy = tf.reduce_mean(tf.reduce_sum(-data['logprobs'], axis=1))
            mean_non_score_reward = tf.reduce_mean(tf.reduce_sum(data['non_score_reward'], axis=1))
            stats = {
                'objective/kl': mean_kl,
                'objective/kl_coef': kl_coef,
                'objective/entropy': mean_entropy,
            }
            for k, v in data['train_stats'].items():
                stats[f'ppo/{k}'] = tf.reduce_mean(v, axis=0)
            for k, v in data['score_stats'].items():
                mean = tf.reduce_mean(v, axis=0)
                stats[f'objective/{k}'] = mean
                stats[f'objective/{k}_total'] = mean + mean_non_score_reward

            stats = utils.FlatStats.from_dict(stats).map_flat(
                partial(utils.mpi_allreduce_mean, comm=self.comm)).as_dict()

            # Add more statistics
            step = tf.train.get_global_step().read_value()
            stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
            steps = step + 1
            stats.update({
                'elapsed/updates': steps,
                'elapsed/steps/serial': steps * hparams.task.response_length,
                'elapsed/steps/total': steps * hparams.ppo.batch_size * hparams.task.response_length,
                'elapsed/episodes': steps * hparams.ppo.batch_size,
            })

            # Time statistics
            total, delta = tf_times()
            stats.update({
                'elapsed/fps': tf.cast(hparams.ppo.batch_size * hparams.task.response_length / delta, tf.int32),
                'elapsed/time': total,
            })
            if ppo_summary_writer:
                record_op = utils.record_stats(
                    stats=stats, summary_writer=ppo_summary_writer, step=step, log_interval=hparams.run.log_interval, name='ppo_stats', comm=self.comm)
            else:
                record_op = tf.no_op()
            return record_op, stats
        self.record_step_stats = record_step_stats

    def print_samples(self, queries, responses, scores, logprobs, ref_logprobs):
        if self.comm.Get_rank() != 0:
            return
        if tf.train.get_global_step().eval() % self.hparams.run.log_interval != 0:
            return

        encoder = self.policy.encoder

        # Log samples
        for i in range(min(3, len(queries))):
            sample_kl = np.sum(logprobs[i] - ref_logprobs[i])
            print(encoder.decode(queries[i][:self.hparams.task.query_length]).replace("\n", "⏎"))
            print(encoder.decode(responses[i]).replace("\n", "⏎"))
            print(f"  score = {scores[i]:+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {scores[i] - self.hparams.rewards.kl_coef * sample_kl:+.2f}")

    def step(self):
        step_started_at = time.time()

        queries = self.sample_queries()
        rollouts = self.policy.respond(queries, length=self.hparams.task.response_length)

        responses = rollouts['responses']
        logprobs = rollouts['logprobs']
        rollouts['queries'] = queries
        ref_logprobs = self.ref_policy.analyze_responses(queries, responses)['logprobs']
        scores, postprocessed_responses, score_stats = self.score_fn(queries, responses)

        rewards, non_score_reward, kl_coef = self.compute_rewards(
            scores=scores,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs)
        rollouts['rewards'] = rewards

        train_stats = self.train(rollouts=rollouts)

        _, stats = self.record_step_stats(
            scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs, non_score_reward=non_score_reward,
            train_stats=train_stats, score_stats=score_stats, kl_coef=kl_coef)

        self.kl_ctl.update(stats['objective/kl'], self.hparams.ppo.batch_size)

        self.print_samples(queries=queries, responses=postprocessed_responses,
                           scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs)

        # Record profiles of the step times
        step = tf.get_default_session().run(tf.train.get_global_step())
        step_time = time.time() - step_started_at
        eps_per_second = float(self.hparams.ppo.batch_size) / step_time
        if self.comm.Get_rank() == 0:
            print(f"[ppo_step {step}] step_time={step_time:.2f}s, "
                  f"eps/s={eps_per_second:.2f}")


    def loss(self, rollouts):
        values = rollouts['values']
        old_logprob = rollouts['logprobs']
        rewards = rollouts['rewards']
        with tf.name_scope('ppo_loss'):
            if self.hparams.ppo.whiten_rewards:
                rewards = utils.whiten(rewards, shift_mean=False)

            lastgaelam = 0
            advantages_reversed = []
            gen_length = self.hparams.task.response_length
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + self.hparams.ppo.gamma * nextvalues - values[:, t]
                lastgaelam = delta + self.hparams.ppo.gamma * self.hparams.ppo.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = tf.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values

            advantages = utils.whiten(advantages)
            advantages = tf.stop_gradient(advantages)  # Shouldn't do anything, but better not to think about it

            outputs = self.policy.analyze_responses_op(rollouts['queries'], rollouts['responses'])

            vpred = outputs['values']
            vpredclipped = tf.clip_by_value(vpred, values - self.hparams.ppo.cliprange_value, values + self.hparams.ppo.cliprange_value)
            vf_losses1 = tf.square(vpred - returns)
            vf_losses2 = tf.square(vpredclipped - returns)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            vf_clipfrac = tf.reduce_mean(tf.cast(tf.greater(vf_losses2, vf_losses1), tf.float32))

            logprob = outputs['logprobs']
            ratio = tf.exp(logprob - old_logprob)
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * tf.clip_by_value(ratio, 1.0 - self.hparams.ppo.cliprange, 1.0 + self.hparams.ppo.cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            pg_clipfrac = tf.reduce_mean(tf.cast(tf.greater(pg_losses2, pg_losses), tf.float32))

            loss = pg_loss + self.hparams.ppo.vf_coef * vf_loss

            entropy = tf.reduce_mean(outputs['entropies'])
            approxkl = .5 * tf.reduce_mean(tf.square(logprob - old_logprob))

            return_mean, return_var = tf.nn.moments(returns, axes=list(range(returns.shape.ndims)))
            value_mean, value_var = tf.nn.moments(values, axes=list(range(values.shape.ndims)))

            stats = dict(
                loss=dict(policy=pg_loss, value=vf_loss, total=loss),
                policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
                returns=dict(mean=return_mean, var=return_var),
                val=dict(vpred=tf.reduce_mean(vpred), error=tf.reduce_mean((vpred - returns) ** 2),
                         clipfrac=vf_clipfrac, mean=value_mean, var=value_var)
            )
            return loss, utils.flatten_dict(stats, sep='/')


def make_score_fn(hparams, score_model):
    padding_token = score_model.padding_token

    postprocess_fn = lm_tasks.postprocess_fn_from_hparams(hparams, padding_token)
    #decorate requires a named function, postprocess_fn can be anonymous
    @utils.graph_function(responses=Schema(tf.int32, (None, None)))
    def postprocess(responses):
        return postprocess_fn(responses)

    filter_fn = lm_tasks.filter_fn_from_hparams(hparams)
    @utils.graph_function(
        responses=Schema(tf.int32, (None, None)),
        rewards=Schema(tf.float32, (None,)))
    def penalize(responses, rewards):
        valid = filter_fn(responses)
        return tf.where(valid, rewards, hparams.penalty_reward_value * tf.ones_like(rewards))

    @utils.graph_function(
        queries=Schema(tf.int32, (None, None)),
        responses=Schema(tf.int32, (None, None))
    )
    def unpenalized_score_fn(queries, responses):
        return score_model.score_fn(queries, responses)

    def score_fn(queries, responses):
        responses = postprocess(responses)
        score = penalize(responses, unpenalized_score_fn(queries, responses))
        return score, responses, dict(score=score)
    score_fn.stat_schemas = dict(score=Schema(tf.float32, (None,)))
    return score_fn



def train(hparams: HParams):
    save_dir = hparams.run.save_dir
    if hparams.rewards.train_new_model:
        assert hparams.task == hparams.rewards.train_new_model.task, f'{hparams.task} != {hparams.rewards.train_new_model.task}'
        hparams.rewards.train_new_model.run.save_dir = save_dir
        train_reward.train(hparams.rewards.train_new_model)
        if 'pytest' in sys.modules:
            hparams.rewards.trained_model = 'test'
        elif save_dir:
            hparams.rewards.trained_model = None if save_dir is None else os.path.join(save_dir, 'reward_model')

    comm = MPI.COMM_WORLD

    with tf.Graph().as_default():
        hyperparams.dump(hparams)

        m = trained_models.TrainedModel(hparams.task.policy.initial_model)
        encoder = m.encoding.get_encoder()
        hyperparams.dump(m.hparams(), name='model_hparams')

        if save_dir:
            if not save_dir.startswith('gs://'):
                os.makedirs(os.path.join(save_dir, 'policy'), exist_ok=True)
            with tf.gfile.Open(os.path.join(save_dir, 'train_policy_hparams.json'), 'w') as f:
                json.dump(hparams.to_nested_dict(), f, indent=2)
            with tf.gfile.Open(os.path.join(save_dir, 'policy', 'hparams.json'), 'w') as f:
                json.dump(m.hparams().to_nested_dict(), f, indent=2)
            with tf.gfile.Open(os.path.join(save_dir, 'policy', 'encoding'), 'w') as f:
                json.dump(m.encoding.name, f, indent=2)
        utils.set_mpi_seed(hparams.run.seed)

        score_model = TrainedRewardModel(hparams.rewards.trained_model, m.encoding, comm=comm)

        ref_policy = Policy(
            m, scope='ref_policy',
            is_root=comm.Get_rank() == 0,
            embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
            temperature=hparams.task.policy.temperature,
            build_respond=False)

        policy = Policy(
            m, scope='policy',
            is_root=comm.Get_rank() == 0,
            embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
            temperature=hparams.task.policy.temperature)

        query_sampler = lm_tasks.make_query_sampler(
            hparams=hparams.task, encoder=encoder, comm=comm,
            batch_size=utils.exact_div(hparams.ppo.batch_size, comm.Get_size()),
        )

        per_rank_minibatch_size = utils.exact_div(hparams.ppo.batch_size, hparams.ppo.nminibatches * comm.Get_size())
        if hparams.ppo.whiten_rewards:
            assert per_rank_minibatch_size >= 8, \
                f"Per-rank minibatch size {per_rank_minibatch_size} is insufficient for whitening"

        global_step = tf.train.get_or_create_global_step()
        increment_global_step = tf.group(global_step.assign_add(1))

        with utils.variables_on_gpu():

            ppo_trainer = PPOTrainer(
                policy=policy, ref_policy=ref_policy, query_sampler=query_sampler,
                score_fn=make_score_fn(hparams.task, score_model=score_model),
                hparams=hparams, comm=comm)

        if comm.Get_rank() == 0 and save_dir:
            print(f"Will save to {save_dir}")
            saver = tf.train.Saver(max_to_keep=20, save_relative_paths=True)
            checkpoint_dir = os.path.join(save_dir, 'policy/checkpoints/model.ckpt')
        else:
            saver = None
            checkpoint_dir = None

        @utils.graph_function()
        def sync_models():
            score_model.ensure_built()
            return utils.variable_synchronizer(comm, vars=score_model.get_params() + ref_policy.get_params() + policy.get_params())

        init_ops = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            summary.summary_writer_initializer_op())

        with utils.mpi_session() as sess:
            init_ops.run()

            sync_models()

            tf.get_default_graph().finalize()

            try:
                while global_step.eval() < nupdates(hparams):
                    ppo_trainer.step()
                    increment_global_step.run()

                    if saver and global_step.eval() % hparams.run.save_interval == 0:
                        saver.save(sess, checkpoint_dir, global_step=global_step)
            finally:
                if saver:
                    saver.save(sess, checkpoint_dir, global_step=global_step)
