import json
import random
import re

import ftfy

from lm_human_preferences.utils import gcs


def tldr_generator(mode, seed=0, shuffle=False, comm=None):
    random.seed(seed)

    if mode == 'test':
        mode = 'valid' # validation set serves as training set, since we don't have access..
    assert mode in ['train', 'valid']

    with open(gcs.download_file_cached(f'gs://lm-human-preferences/tldr/{mode}-subset.json', comm=comm)) as f:
        datas = json.load(f)

    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    for data in datas:
        text = data['content']
        text = ftfy.fix_text(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        yield text
