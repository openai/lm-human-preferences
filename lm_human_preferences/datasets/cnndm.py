import hashlib
import os
import random
import re

import ftfy

from lm_human_preferences.utils import gcs

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line=="":
        return line
    if line[-1] in END_TOKENS:
        return line
    # print line[-1]
    return line + "."

def get_art_abs(story_file):
    lines = read_text_file(story_file)
    # lines = [fix_missing_period(line) for line in lines]
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)
    article = '\n\n'.join(article_lines)

    # Make abstract into a single string, putting <s> and </s> tags around the sentences
    highlights = [fix_missing_period(sent) for sent in highlights]
    # abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])
    # abstract = ' '.join(highlights)
    return article, highlights

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

def get_path_of_url(url):
    if 'dailymail.co.uk' in url or 'mailonsunday.ie' in url or 'lib.store.yahoo.net' in url:
        site = 'dailymail'
    else:
        assert 'cnn.com' in url or 'cnn.hk' in url, url
        site = 'cnn'
    url_hash = hashhex(url.encode('utf-8'))
    return f'{site}/stories/{url_hash}.story'

def clean_up_start(text):
    if text[:2] == 'By':
        text = '\n'.join(text.split('\n')[2:])
    text = re.split(r'\(CNN\) +--', text)[-1]
    text = re.split(r"\(CNN\)", text[:100])[-1]+text[100:]
    text = re.sub(r"^and \w+\n", "", text)
    text = re.split(r".*UPDATED:\s+[0-9]{2}:[0-9]{2}.*[2011|2012|2013|2014|2015]", text)[-1]
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    return text.strip()

def cnndm_generator(mode, seed=0, shuffle=False, comm=None):
    # data originally from https://github.com/abisee/cnn-dailymail
    if mode == 'valid':
        mode = 'val'
    with open(gcs.download_file_cached(f'gs://lm-human-preferences/datasets/cnndm/url_lists/all_{mode}.txt', comm=comm)) as f:
        urls = [line.strip() for line in f]
    if shuffle:
        random.seed(seed)
        random.shuffle(urls)
    # if n_eval > 0:
    #     urls = urls[:n_eval]

    urls_dir = gcs.download_directory_cached(f'gs://lm-human-preferences/datasets/cnndm/cache_{mode}', comm=comm)

    for i, url in enumerate(urls):
        path = os.path.join(urls_dir, get_path_of_url(url))
        text = open(path).read()
        text = clean_up_start(text)
        text = ftfy.fix_text(text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.split('@highlight')[0].strip()
        yield text
        # _, ref_sents = get_art_abs(path)
