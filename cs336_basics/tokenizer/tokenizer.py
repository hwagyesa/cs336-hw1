#!/usr/bin/env python
# -*- coding: utf-8 -*-

import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytest
import regex as re

# Tokenizer class and tests


def pretokenizer_gpt2(text: str) -> list[str]:
    # GPT2 pretokenizer
    # https://github.com/openai/tiktoken/pull/234/files
    pat_str = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    return re.findall(pat_str, text)


@dataclass
class BPENaive:
    """BPE-based utf8 tokenizer."""

    corpus_path: Path = Path(".")
    max_vocab_size: int = 1024
    pretokenizer: Callable = pretokenizer_gpt2
    special_tokens: List[str] = field(default_factory=list)
    special_tokens_dict: Dict[str, int] = field(default_factory=dict, init=False)
    vocab: Dict[int, bytes] = field(default_factory=dict, init=False)
    merges: list[tuple[int, int]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._map_special_tokens()
        self._train(self.corpus_path)

    def _map_special_tokens(self):
        # Scheme: split the string on special tokens. Tokenize these manually.
        # Then operate normally on the rest. (no cou nting on special tokens)
        # NOTE: Assume "valid" special tokens (no substring issues)
        for i, token in enumerate(self.special_tokens):
            self.special_tokens_dict[token] = i
            self.vocab[i] = token.encode("utf8")

    def _split_on_special_tokens(self, text: str, training=False) -> List[str]:
        """training=True strips the special tokens. False preserves them (for encoding)."""

        special_token_pat = "|".join(map(re.escape, self.special_tokens))
        if training:
            return re.split(special_token_pat, text)
        else:
            return re.split("(" + special_token_pat + ")", text)

    def _train(self, corpus_path: Path):
        # Initialize with bytes.
        for i in range(256):
            self.vocab[i + len(self.special_tokens)] = bytes([i])
        # Get corpus content
        with open(corpus_path, "r") as file:
            corpus = file.read()
        # Strip special tokens.
        corpus = "".join(self._split_on_special_tokens(corpus, training=True))
        # Pre-tokenize and create count table
        text = self.pretokenizer(corpus)
        # BUG: we count overlapping tokens, but should probably disregard (sentencepiece...)
        count_dict_pretoken: Dict[Tuple[int, ...], int] = {}
        # count_dict_pair = Dict[tuple[int, int], int] = {}
        for pretoken in text:
            token = tuple(n + len(self.special_tokens) for n in pretoken.encode("utf8"))
            if token not in count_dict_pretoken:
                count_dict_pretoken[token] = 1
            else:
                count_dict_pretoken[token] += 1
        while len(self.vocab) <= self.max_vocab_size:
            # count frequencies
            count_dict: Dict[tuple[int, int], int] = {}
            for pretoken in count_dict_pretoken.keys():
                for pair in zip(pretoken, pretoken[1:]):
                    if pair not in count_dict:
                        count_dict[pair] = count_dict_pretoken[pretoken]
                    else:
                        count_dict[pair] += count_dict_pretoken[pretoken]
            # Most frequent scan
            most_frequent_pair = None
            max_count = -1
            for pair in sorted(count_dict.keys()):
                count = count_dict[pair]
                if count >= max_count:
                    most_frequent_pair = pair
                    max_count = count
            # Log merge and update vocab
            if most_frequent_pair is None:
                # Nothing left to merge.
                break
            self.merges.append(most_frequent_pair)
            with pytest.raises(KeyError):
                print(self.vocab[len(self.vocab)])
            self.vocab[len(self.vocab)] = (
                self.vocab[most_frequent_pair[0]] + self.vocab[most_frequent_pair[1]]
            )
            # Update the pretoken count dict (manual merge)
            # PERF: conversion to/from tuple is wasteful (array.array not hashable so need to...)
            for pretoken in list(count_dict_pretoken):
                new_key = array.array("I", pretoken)
                new_key = BPENaive._merge(
                    new_key, most_frequent_pair, len(self.vocab) - 1
                )
                new_key = tuple(new_key)
                if new_key != pretoken:
                    count_dict_pretoken[new_key] = count_dict_pretoken[pretoken]
                    del count_dict_pretoken[pretoken]

    @staticmethod
    def _merge(
        token_seq: array.ArrayType, merge_pair: tuple[int, int], new_token: int
    ) -> array.ArrayType:
        ptr = 0
        while ptr < len(token_seq) - 1:
            byte = token_seq[ptr]
            next_byte = token_seq[ptr + 1]
            if (byte, next_byte) == merge_pair:
                token_seq[ptr] = new_token
                token_seq[ptr + 1 :] = token_seq[ptr + 2 :]
                # token_seq = token_seq[:-1]
            else:
                ptr += 1
        return token_seq

    def encode(self, text: str) -> list[int]:
        # The encoding process involves converting to utf8, then applying the merges.
        # We also need to start by treating special characters in a special way.
        text_split_special = self._split_on_special_tokens(text, training=False)
        text_encoded = []
        for segment in text_split_special:
            if segment in self.special_tokens:
                token = self.special_tokens_dict[segment]
                text_encoded.append(token)
            else:
                segment_encoded = array.array(
                    "I",
                    (
                        byte + len(self.special_tokens)
                        for byte in segment.encode("utf8")
                    ),
                )
                token = 256 + len(self.special_tokens)
                for merge in self.merges:
                    segment_encoded = BPENaive._merge(segment_encoded, merge, token)
                    token += 1
                text_encoded += segment_encoded
        return text_encoded

    def decode(self, tokens: list[int]) -> str:
        # The decoding process simply involves the vocab lookup.
        output = bytes([])
        for token in tokens:
            output += self.vocab[token]
        return output.decode("utf8")


def test_BPE_naive():
    corpus_path = Path("./test_data/test.txt")
    vocab_size = 512  # 'initial' size is 256 (bytes)
    tokenizer = BPENaive(corpus_path, vocab_size, special_tokens=["<|STOP|>"])

    test_str = (
        "Hello, world! This is a test.<|STOP|>여러분들, 안녕하세요? 12,34 1 -- 3 #$@$)@"
        * 200
    )

    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_str


if __name__ == "__main__":
    # Some benchmarking
    pass
