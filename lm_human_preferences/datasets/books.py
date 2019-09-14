import json
import random

from lm_human_preferences.utils import gcs


def books_generator(mode, seed=0, shuffle=False, comm=None):
    datas = [
        json.loads(line) for line in
        open(gcs.download_file_cached(f'gs://lm-human-preferences/datasets/book_passages/{mode}.jsonl', comm=comm))
    ]
    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    for x in datas:
        yield x
