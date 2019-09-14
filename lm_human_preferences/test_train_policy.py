#!/usr/bin/env python3

import tempfile
from lm_human_preferences import train_policy

def hparams_for_test():
    hparams = train_policy.HParams()
    hparams.ppo.batch_size = 8
    hparams.noptepochs = 1
    hparams.task.policy.initial_model = 'test'
    hparams.task.query_length = 2
    hparams.task.response_length = 3
    hparams.task.query_dataset = 'test'
    hparams.rewards.trained_model = 'test'
    hparams.ppo.total_episodes = 8
    hparams.run.log_interval = 1

    return hparams


def train_policy_test(override_params):
    hparams = hparams_for_test()
    hparams.override_from_dict(override_params)
    hparams.validate()
    train_policy.train(hparams=hparams)


def test_truncation():
    train_policy_test({
        'task.truncate_token': 13,
        'task.truncate_after': 2,
    })

def test_defaults():
    train_policy_test({})

def test_affixing():
    train_policy_test({
        'task.query_prefix': 'a',
        'task.query_suffix': 'b'
    })

def test_adaptive_kl():
    train_policy_test({
        'rewards.trained_model': 'test', # not sure why needed
        'rewards.adaptive_kl': 'on',
        'rewards.adaptive_kl.target': 3.0,
        'rewards.adaptive_kl.horizon': 100,
    })

def test_save():
    train_policy_test({
        'run.save_dir': tempfile.mkdtemp() ,
        'run.save_interval': 1
    })

def test_reward_training():
    train_policy_test({
        'rewards.trained_model': None,
        'rewards.train_new_model': 'on',
        'rewards.train_new_model.task.policy.initial_model': 'test',
        'rewards.train_new_model.task.query_length': 2,
        'rewards.train_new_model.task.response_length': 3,
        'rewards.train_new_model.task.query_dataset': 'test',
        'rewards.train_new_model.labels.source': 'test',
        'rewards.train_new_model.labels.num_train': 16,
        'rewards.train_new_model.batch_size': 8,
        'rewards.train_new_model.labels.type': 'best_of_4',
    })
