import random
from typing import Dict

import tensorflow as tf

from lm_human_preferences.datasets.books import books_generator
from lm_human_preferences.datasets.cnndm import cnndm_generator
from lm_human_preferences.datasets.tldr import tldr_generator

_registry: Dict[str, "Dataset"] = {}

class Dataset:
    def __init__(
            self,
            name,
            *,
            generator=None,
    ):
        global _registry
        assert name not in _registry
        _registry[name] = self

        self.name = name

        self.generator = generator

    def tf_dataset(
            self,
            sequence_length,
            *,
            mode,
            encoder=None,
            seed=0,
            comm=None,
            shuffle=True,
            repeat_count=None,  # Defaults to infinite repeat
            # trims so that it starts right after start token
            start_token=None,
            # trims off last end_token
            end_token=None,
            padding_token=None,
    ):
        if padding_token is None:
            padding_token = encoder.padding_token
        def _generator():
            inner_gen = self.generator(mode, seed=seed, shuffle=shuffle, comm=comm)
            for text in inner_gen:
                tokens = encoder.encode(text)
                if start_token is not None:
                    try:
                        first_index = tokens.index(start_token)+1
                        if first_index < len(tokens):
                            tokens = tokens[first_index:]
                    except:
                        continue

                tokens = tokens[:sequence_length]

                if end_token is not None:
                    try:
                        last_index = len(tokens)-tokens[::-1].index(end_token)
                        tokens = tokens[:last_index]
                    except:
                        continue

                if len(tokens) < sequence_length:
                    tokens = tokens + [padding_token] * (sequence_length - len(tokens))

                assert len(tokens) == sequence_length

                yield dict(tokens=tokens)

        tf_dataset = tf.data.Dataset.from_generator(
            _generator,
            output_types=dict(tokens=tf.int32),
            output_shapes=dict(tokens=(sequence_length,)),
        )
        tf_dataset = tf_dataset.repeat(repeat_count)

        if comm is not None:
            num_shards = comm.Get_size()
            shard_idx = comm.Get_rank()
            if num_shards > 1:
                assert seed is not None
                tf_dataset = tf_dataset.shard(num_shards, shard_idx)

        return tf_dataset


def get_dataset(name) -> Dataset:
    global _registry
    return _registry[name]

CnnDm = Dataset(
    "cnndm",
    generator=cnndm_generator,
)

Tldr = Dataset(
    "tldr",
    generator=tldr_generator,
)

Books = Dataset(
    "books",
    generator=books_generator,
)

def test_generator(mode, seed=0, shuffle=False, comm=None):
    while True:
        yield ''.join([random.choice('abcdefghijklmnopqrstuvwxyz.') for _ in range(40)])

Test = Dataset(
    "test",
    generator=test_generator
)


"""
import tensorflow as tf
from lm_human_preferences.language.datasets import Books as ds
from lm_human_preferences.language.encodings import Main as encoding

e = encoding.get_encoder()
x = ds.tf_dataset(16, mode='test', encoder=e)
op = x.make_one_shot_iterator().get_next()
s = tf.Session()

while True:
    print(e.decode(s.run(op)['tokens']))
    input()
"""
