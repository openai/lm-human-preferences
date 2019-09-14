"""Byte pair encoding utilities"""

import json
import os
from functools import lru_cache

import tensorflow as tf
import regex as re

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class ReversibleEncoder:
    def __init__(self, encoder, bpe_merges, errors="replace", eot_token=None):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.eot_token = eot_token
        self.cache = {}
        self.padding_token = len(encoder) + 2 # +2 unnecessary, for historical reasons
        self.decoder[self.padding_token] = ''

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens, pretty=False):
        del pretty
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text


def read_file(path):
    with tf.gfile.Open(path, "rb") as fh:
        return fh.read()


class Encoding:
    def __init__(
            self,
            name,
            *,
            n_vocab=0,
            eot_token=None,
            encoder_path="encoder.json",
            bpe_path="vocab.bpe",
            base_path=None,
    ):
        self.name = name
        self.eot_token = eot_token
        self.n_vocab = n_vocab

        if base_path is None:
            base_path = os.path.join("gs://gpt-2/encodings", name)

        self.base_path = base_path
        if name != "test":
            self.encoder_path = os.path.join(self.base_path, encoder_path)
            self.bpe_path = os.path.join(self.base_path, bpe_path)

    def get_encoder(self):
        if self.name == "test":
            vocab = "abcdefghijklmnopqrstuvwxyz."
            assert len(vocab) == self.n_vocab

            class TestEncoder(ReversibleEncoder):
                def __init__(self):
                    super().__init__(encoder={w: i for i, w in enumerate(vocab)}, bpe_merges=list())
                    self.padding_token = len(vocab)
                def encode(self, text):
                    return [self.encoder.get(x, len(vocab) - 1) for x in text]
                def decode(self, tokens, pretty=False):
                    return ''.join([self.decoder.get(t, '<unk>') for t in tokens])

            return TestEncoder()

        encoder_dict = json.loads(read_file(self.encoder_path).decode())
        bpe_data = read_file(self.bpe_path).decode()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
        assert len(encoder_dict) == self.n_vocab
        encoder = ReversibleEncoder(encoder=encoder_dict, bpe_merges=bpe_merges, eot_token=self.eot_token)
        assert encoder.padding_token >= self.n_vocab
        return encoder


Main = Encoding("main", n_vocab=50257, eot_token=50256)

Test = Encoding("test", n_vocab=27, eot_token=26)
