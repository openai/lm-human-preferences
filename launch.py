#!/usr/bin/env python3

from lm_human_preferences.utils import launch
from lm_human_preferences.utils.combos import bind, combos, each, label, options_shortdesc, bind_nested
from lm_human_preferences import train_policy, train_reward


books_task = combos(
    bind('query_length', 64),
    bind('query_dataset', 'books'),
    bind('response_length', 24),
    bind('start_text', '.'), # Start the context at the beginning of a sentence
    bind('end_text', '.'), # End the context at the end of a sentence.
    bind('truncate_token', 13), # Encoding of '.' -- end completions at the end of a sentence.
    bind('truncate_after', 16), # Make sure completions are at least 16 tokens long.

    bind('policy.temperature', 0.7),
    bind('policy.initial_model', '124M'),
)

summarize_cnndm_task = combos(
    bind('query_prefix', 'Article:\n\n'),
    bind('query_suffix', '\n\nTL;DR:'),
    bind('end_text', '\n'),
    bind('query_dataset', 'cnndm'),
    bind('query_length', 500),
    bind('response_length', 75),
    bind('start_text', None),
    bind('truncate_after', 55),
    bind('truncate_token', 198),  # '\n'

    bind('policy.temperature', 0.5),
    bind('policy.initial_model', '124M'),
)

summarize_tldr_task = combos(
    bind('query_suffix', '\n\nTL;DR:'),
    bind('query_dataset', 'tldr'),
    bind('query_length', 500),
    bind('response_length', 75),
    bind('start_text', None),
    bind('truncate_after', 55),
    bind('truncate_token', 198),  # '\n'

    bind('policy.temperature', 0.7),
    bind('policy.initial_model', '124M'),
)

def get_train_reward_experiments():
    _shared = combos(
        bind('labels.type', 'best_of_4'),
        bind('normalize_after', True),
        bind('normalize_before', True),
        bind('normalize_samples', 256),
    )


    _books_task = combos(
        bind_nested('task', books_task),
        _shared,
        bind('batch_size', 32),
        bind('lr', 5e-5),
        bind('rollout_batch_size', 512),
    )

    sentiment = combos(
        _books_task,

        bind('labels.source', 'gs://lm-human-preferences/labels/sentiment/offline_5k.json'),
        bind('labels.num_train', 4_992),
        bind('run.seed', 1)
    )


    descriptiveness = combos(
        _books_task,

        bind('labels.source', 'gs://lm-human-preferences/labels/descriptiveness/offline_5k.json'),
        bind('labels.num_train', 4_992),
        bind('run.seed', 1)
    )

    cnndm = combos(
        bind_nested('task', summarize_cnndm_task),
        _shared,

        # bind('labels.source', 'gs://lm-human-preferences/labels/cnndm/offline_60k.json'),
        # bind('labels.num_train', 60_000),
        bind('labels.source', 'gs://lm-human-preferences/labels/cnndm/online_45k.json'),
        bind('labels.num_train', 46_000),

        bind('batch_size', 2 * 8),
        bind('lr', 2.5e-5),
        bind('rollout_batch_size', 128),
        bind('run.seed', 1)
    )

    tldr = combos(
        bind_nested('task', summarize_tldr_task),
        _shared,

        # bind('labels.source', 'gs://lm-human-preferences/labels/tldr/offline_60k.json'),
        # bind('labels.num_train', 60_000),
        bind('labels.source', 'gs://lm-human-preferences/labels/tldr/online_45k.json'),
        bind('labels.num_train', 46_000),

        bind('batch_size', 2 * 8),
        bind('lr', 2.5e-5),
        bind('rollout_batch_size', 128),
        bind('run.seed', 1)
    )

    return locals()


def get_experiments():
    train_reward_experiments = get_train_reward_experiments()

    _books_task = combos(
        bind_nested('task', books_task),

        bind('ppo.lr', 1e-5),
        bind('ppo.total_episodes', 1_000_000),
        bind('ppo.batch_size', 512),
    )

    sentiment = combos(
        _books_task,
        bind('rewards.kl_coef', 0.15),
        bind('rewards.adaptive_kl', 'on'),
        bind('rewards.adaptive_kl.target', 6.0),

        bind('rewards.train_new_model', 'on'),
        bind_nested('rewards.train_new_model', train_reward_experiments['sentiment']),
        # bind('rewards.trained_model', '/your/directory/here/reward_model/'),

        bind('run.seed', 1)
    )

    descriptiveness = combos(
        _books_task,
        bind('rewards.kl_coef', 0.15),
        bind('rewards.adaptive_kl', 'on'),
        bind('rewards.adaptive_kl.target', 6.0),

        bind('rewards.train_new_model', 'on'),
        bind_nested('rewards.train_new_model', train_reward_experiments['descriptiveness']),
        # bind('rewards.trained_model', '/your/directory/here/reward_model/'),

        bind('run.seed', 1)
    )

    cnndm = combos(
        bind_nested('task', summarize_cnndm_task),

        bind('rewards.train_new_model', 'on'),
        bind_nested('rewards.train_new_model', train_reward_experiments['cnndm']),
        # bind('rewards.trained_model', '/your/directory/here/reward_model/'),

        bind('ppo.total_episodes', 1_000_000),
        bind('ppo.lr', 2e-6),
        bind('rewards.kl_coef', 0.01),
        # bind('rewards.adaptive_kl', 'on'),
        # bind('rewards.adaptive_kl.target', 18.0),
        bind('ppo.batch_size', 32),
        bind('rewards.whiten', False),

        bind('run.seed', 1)
    )

    tldr = combos(
        bind_nested('task', summarize_tldr_task),

        bind('rewards.train_new_model', 'on'),
        bind_nested('rewards.train_new_model', train_reward_experiments['tldr']),
        # bind('rewards.trained_model', '/your/directory/here/reward_model/'),

        bind('ppo.total_episodes', 1_000_000),
        bind('ppo.lr', 2e-6),
        bind('rewards.kl_coef', 0.03), # 0.01 too low
        # bind('rewards.adaptive_kl', 'on'),
        # bind('rewards.adaptive_kl.target', 18.0),
        bind('ppo.batch_size', 32),
        bind('rewards.whiten', False),

        bind('run.seed', 1)
    )

    return locals()


def launch_train_policy(exp, name, dry_run=False, mpi=8, mode='local', save_dir='/tmp/save/train_policy', **extra_hparams):
    experiment_dict = get_experiments()
    try:
        trials = experiment_dict[exp]
    except KeyError:
        raise ValueError(f"Couldn't find experiment '{exp}'")

    launch.launch_trials(
        name, fn=train_policy.train, trials=trials, mpi=mpi, mode=mode, save_dir=save_dir,
        hparam_class=train_policy.HParams, extra_hparams=extra_hparams, dry_run=dry_run)


def launch_train_reward(exp, name, dry_run=False, mpi=8, mode='local', save_dir='/tmp/save/train_reward', **extra_hparams):
    experiment_dict = get_train_reward_experiments()
    try:
        trials = experiment_dict[exp]
    except KeyError:
        raise ValueError(f"Couldn't find experiment '{exp}'")

    launch.launch_trials(
        name, fn=train_reward.train, trials=trials, mpi=mpi, mode=mode, save_dir=save_dir,
        hparam_class=train_reward.HParams, extra_hparams=extra_hparams, dry_run=dry_run)


if __name__ == '__main__':
    launch.main(dict(
        train_policy=launch_train_policy,
        train_reward=launch_train_reward
    ))
