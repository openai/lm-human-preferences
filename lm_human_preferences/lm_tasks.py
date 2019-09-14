from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf

from lm_human_preferences.language import datasets
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import hyperparams


@dataclass
class PolicyHParams(hyperparams.HParams):
    temperature: float = 1.0
    initial_model: str = None

@dataclass
class TaskHParams(hyperparams.HParams):
    # Query params
    query_length: int = None
    query_dataset: str = None
    query_prefix: str = ''
    query_suffix: str = ''
    start_text: Optional[str] = '.'
    end_text: Optional[str] = None

    # Response params
    response_length: int = None

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Optional[int] = None
    truncate_after: int = 0
    penalty_reward_value: int = -1

    policy: PolicyHParams = field(default_factory=PolicyHParams)

#returns a postprocessing function
#it is applied to responses before they are scored
#central example: replace all tokens after truncate_token with padding_token
def postprocess_fn_from_hparams(hparams: TaskHParams, padding_token: int):
    def get_mask(responses, truncate_token, truncate_after):
        # We want to truncate at the first occurrence of truncate_token that appears at or after
        # position truncate_after in the responses
        mask = tf.cast(tf.equal(responses, truncate_token), tf.int32)
        mask = tf.concat([tf.zeros_like(mask)[:,:truncate_after], mask[:,truncate_after:]], axis=1)
        return tf.cast(tf.cumsum(mask, axis=1) - mask, tf.bool)
    if hparams.truncate_token is not None:
        def truncate(responses):
            mask = get_mask(responses, hparams.truncate_token, hparams.truncate_after)
            return tf.where(mask, padding_token * tf.ones_like(responses), responses)
        return truncate
    else:
        return lambda responses: responses

#returns a filter function
#responses not passing that function will receive a low (fixed) score
#only query humans on responses that pass that function
#central example: ensure that the sample contains truncate_token
def filter_fn_from_hparams(hparams: TaskHParams):
    def filter(responses):
        if hparams.truncate_token is not None:
            matches_token = tf.equal(responses[:, hparams.truncate_after:], hparams.truncate_token)
            return tf.reduce_any(matches_token, axis=-1)
        else:
            return tf.ones(tf.shape(responses)[0], dtype=tf.bool)
    return filter


def query_formatter(hparams: TaskHParams, encoder):
    """Turns a query into a context to feed to the language model

    NOTE: Both of these are lists of tokens
    """
    def query_formatter(queries):
        batch_size = tf.shape(queries)[0]
        prefix_tokens = tf.constant(encoder.encode(hparams.query_prefix), dtype=tf.int32)
        tiled_prefix = utils.expand_tile(prefix_tokens, batch_size, axis=0)
        suffix_tokens = tf.constant(encoder.encode(hparams.query_suffix), dtype=tf.int32)
        tiled_suffix = utils.expand_tile(suffix_tokens, batch_size, axis=0)
        return tf.concat([tiled_prefix, queries, tiled_suffix], 1)
    return query_formatter


def make_query_sampler(*, hparams: TaskHParams, encoder, batch_size: int, mode='train', comm=None):
    if hparams.start_text:
        start_token, = encoder.encode(hparams.start_text)
    else:
        start_token = None

    if hparams.end_text:
        end_token, = encoder.encode(hparams.end_text)
    else:
        end_token = None

    data = datasets.get_dataset(hparams.query_dataset).tf_dataset(
        sequence_length=hparams.query_length, mode=mode, comm=comm, encoder=encoder,
        start_token=start_token, end_token=end_token,
    )
    data = data.map(lambda d: tf.cast(d['tokens'], tf.int32))
    data = data.batch(batch_size, drop_remainder=True)

    context_iterator = data.make_one_shot_iterator()

    def sampler(scope=None):
        with tf.name_scope(scope, 'sample_corpus'):
            context_tokens = context_iterator.get_next()
            return dict(tokens=context_tokens)
    return sampler
