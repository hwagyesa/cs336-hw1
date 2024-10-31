#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heapq
import operator
import pprint
from dataclasses import dataclass, field
from functools import reduce, total_ordering
from pathlib import Path
from types import NoneType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pytest
import regex as re
from sortedcontainers import SortedDict, SortedSet

# Tokenizer class and tests


@dataclass
class Node:
    value: bytes
    rank: int
    next: Optional["Node"] = None
    prev: Optional["Node"] = None

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"Node(value={self.value})"

    def __hash__(self) -> int:
        # Use object's id as hash - nodes are only equal if they're the same object
        return id(self)

    def __eq__(self, other: Any) -> bool:
        # Nodes are equal only if they're the same object
        if not isinstance(other, Node):
            return NotImplemented
        return id(self) == id(other)


@dataclass
class DLinkedList:
    head: Optional[Node] = None

    def append(self, value: bytes) -> None:
        if not self.head:
            self.head = Node(value, 0)
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = Node(value, current.rank + 1)
        current.next.prev = current

    def __str__(self) -> str:
        return " <-> ".join(str(node.value) for node in self)

    def __iter__(self) -> Iterator[Node]:
        current = self.head
        while current:
            yield current
            current = current.next


class BPEEncoder:
    def __init__(self):
        self.priority_queue: SortedDict[
            tuple[int, tuple[bytes, bytes]], SortedSet[Node]
        ] = SortedDict()
        self.pair_counts: Dict[tuple[bytes, bytes], tuple[int, SortedSet[Node]]] = {}

    def initialize_from_string(self, text: str) -> DLinkedList:
        string = DLinkedList()
        for char in text:
            string.append(char.encode("utf8"))

        self._initialize_counts(string)
        return string

    def _initialize_counts(self, string: DLinkedList) -> None:
        current = string.head
        while current and current.next:
            pair = (current.value, current.next.value)
            count, ptrs = self.pair_counts.get(
                pair, (0, SortedSet(key=lambda x: x.rank))
            )
            ptrs.add(current)
            self.pair_counts[pair] = (count + 1, ptrs)
            # Move to the next pair, checking for overlap (we don't double count)
            if (
                current.value == current.next.value
                and current.next.next
                and current.next.value == current.next.next.value
            ):
                current = current.next.next
            else:
                current = current.next

        for pair, (count, ptrs) in self.pair_counts.items():
            self.priority_queue[(count, pair)] = ptrs

    def _update_pair_count(
        self, pair: tuple[bytes, bytes], node: Node, count_delta: int
    ) -> None:
        """Helper to update counts for a pair in both data structures"""
        if pair not in self.pair_counts:
            if count_delta > 0:
                self.pair_counts[pair] = (
                    count_delta,
                    SortedSet([node], key=lambda x: x.rank),
                )
                self.priority_queue[(count_delta, pair)] = SortedSet(
                    [node], key=lambda x: x.rank
                )
            return

        old_count, ptrs = self.pair_counts[pair]
        del self.priority_queue[(old_count, pair)]

        new_count = old_count + count_delta
        if count_delta > 0:
            ptrs.add(node)
            # Perform overlap check
            if (
                node.prev
                and node.prev in ptrs
                and node.prev.prev
                and node.prev.prev in ptrs
            ):
                # Overlap detected.
                # POSSIBLE BUG: We'll assume count_delta is the same as for previous pair. I think ok...
                ptrs.remove(node.prev)
                new_count -= count_delta
        else:
            ptrs.remove(node)

        if new_count > 0:
            self.pair_counts[pair] = (new_count, ptrs)
            self.priority_queue[(new_count, pair)] = ptrs
        else:
            del self.pair_counts[pair]

    def merge_most_frequent(self) -> None:
        """Merge the most frequent pair and update data structures"""
        if not self.priority_queue:
            return

        (count, pair), nodes = self.priority_queue.popitem()
        del self.pair_counts[pair]

        for node in nodes:
            if not node.next:  # Skip if no next node to merge with
                continue

            # Update counts for affected pairs
            if node.prev:
                self._update_pair_count((node.prev.value, node.value), node.prev, -1)

            # If there's a pair after the one we're merging, update its count
            if node.next.next:
                after_pair = (node.next.value, node.next.next.value)
                self._update_pair_count(after_pair, node.next, -1)

            # Perform merge
            node.value += node.next.value
            old_next = node.next.next
            node.next = old_next
            if old_next:
                old_next.prev = node

            # Add new pairs
            if node.prev:
                self._update_pair_count((node.prev.value, node.value), node.prev, 1)
            if node.next:
                self._update_pair_count((node.value, node.next.value), node, 1)


def print_state(encoder: BPEEncoder, string: DLinkedList, iteration: int) -> None:
    """Helper function to print the current state of the encoder"""
    print(f"\nIteration {iteration}")
    print(f"Current string: {string}")
    print("Priority Queue:")
    for (count, pair), nodes in encoder.priority_queue.items():
        print(f"  Count: {count}, Pair: {pair}, Nodes: {len(nodes)}")


def test_bpe_encoder():
    """Test the BPE encoder with the example string"""
    test_str = "peter piper picked a peck of pickled peppers"
    encoder = BPEEncoder()
    string = encoder.initialize_from_string(test_str)

    print("Initial state:")
    print_state(encoder, string, 0)

    # Perform several merges
    for i in range(1, 6):  # Do 5 merges
        encoder.merge_most_frequent()
        print_state(encoder, string, i)

    # Test edge cases
    print("\nTesting edge cases:")

    # Test single character
    encoder = BPEEncoder()
    string = encoder.initialize_from_string("a")
    print("\nSingle character:")
    print_state(encoder, string, 0)
    encoder.merge_most_frequent()  # Should handle gracefully

    # Test repeated characters
    encoder = BPEEncoder()
    string = encoder.initialize_from_string("aaaaaa")
    print("\nRepeated characters:")
    print_state(encoder, string, 0)
    encoder.merge_most_frequent()
    print_state(encoder, string, 1)

    # Test overlapping pairs
    encoder = BPEEncoder()
    string = encoder.initialize_from_string("aaaa")
    print("\nOverlapping pairs:")
    print_state(encoder, string, 0)
    encoder.merge_most_frequent()
    print_state(encoder, string, 1)
    print("hi")


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
    vocab: Dict[int, bytes] = field(default_factory=dict)
    merges: list[tuple[bytes, bytes]] = field(default_factory=list)
    vocab_inverse: Dict[bytes, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if len(self.vocab) == 0:
            # We allow for vocab/merges to be passed in from a pretrained tokenizer.
            self._map_special_tokens()
            self._train(self.corpus_path)
        # BUG: could be a bug, not being careful...
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}

    def _map_special_tokens(self):
        # Scheme: split the string on special tokens. Tokenize these manually.
        # Then operate normally on the rest. (no cou nting on special tokens)
        # NOTE: Assume "valid" special tokens (no substring issues)
        for i, token in enumerate(self.special_tokens):
            # self.special_tokens_dict[token] = i
            self.vocab[i] = token.encode("utf8")

    def _split_on_special_tokens(self, text: str, training=False) -> List[str]:
        """training=True strips the special tokens. False preserves them (for encoding)."""

        if len(self.special_tokens) == 0:
            # short circuit: empty regex breaks everything up
            return [text]
        patterns = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pat = "|".join(map(re.escape, patterns))
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
        # count_dict_pretoken: Dict[Tuple[int, ...], int] = {}
        count_dict_pretoken: Dict[Tuple[bytes, ...], int] = {}
        # count_dict_pair = Dict[tuple[int, int], int] = {}
        for pretoken in text:
            token = tuple(bytes([b]) for b in pretoken.encode("utf8"))
            if token not in count_dict_pretoken:
                count_dict_pretoken[token] = 1
            else:
                count_dict_pretoken[token] += 1
        while len(self.vocab) < self.max_vocab_size:
            # count frequencies
            count_dict: Dict[tuple[bytes, bytes], int] = {}
            for pretoken in count_dict_pretoken.keys():
                for pair in zip(pretoken, pretoken[1:]):
                    if pair not in count_dict:
                        count_dict[pair] = count_dict_pretoken[pretoken]
                    else:
                        count_dict[pair] += count_dict_pretoken[pretoken]
            # Most frequent scan
            most_frequent_pair = None
            max_count = -1
            for pair in sorted(
                count_dict.keys()
            ):  # Guarantees lexicographic merge ordering
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
            self.vocab[len(self.vocab)] = reduce(operator.add, most_frequent_pair)
            # Update the pretoken count dict (manual merge)
            # PERF: conversion to/from tuple is wasteful (array.array not hashable so need to...)
            for pretoken in list(count_dict_pretoken):
                new_key = list(pretoken)  # can't avoid this?
                new_key = BPENaive._merge(new_key, most_frequent_pair)
                new_key = tuple(new_key)  # keys need to be mutable...
                if new_key != pretoken:
                    count_dict_pretoken[new_key] = count_dict_pretoken[pretoken]
                    del count_dict_pretoken[pretoken]

    @staticmethod
    def _merge(token_seq: List[bytes], merge_pair: tuple[bytes, bytes]) -> List[bytes]:
        ptr = 0
        while ptr < len(token_seq) - 1:
            byte_seq = token_seq[ptr]
            next_byte_seq = token_seq[ptr + 1]
            if (byte_seq, next_byte_seq) == merge_pair:
                token_seq[ptr] = byte_seq + next_byte_seq
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
                segment_encoded = segment.encode("utf8")
                token = self.vocab_inverse[segment_encoded]
                text_encoded.append(token)
            else:
                segment_pretokenized = self.pretokenizer(segment)
                for segment in segment_pretokenized:
                    segment_encoded = list(bytes([b]) for b in segment.encode("utf8"))
                    for merge in self.merges:
                        segment_encoded = BPENaive._merge(segment_encoded, merge)
                    tokens = [
                        self.vocab_inverse[byte_seq] for byte_seq in segment_encoded
                    ]
                    text_encoded += tokens
        return text_encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, tokens: list[int]) -> str:
        # The decoding process simply involves the vocab lookup.
        output = bytes([])
        for token in tokens:
            output += self.vocab[token]
        return output.decode("utf8", errors="replace")


# TODO: Fast training implementation: better data structures for merges.
# TODO: Fast encoding implementation: support chunking, possible merge optimizations.
# Chunking notes.
# Chunk with (forward) overlaps (one-side).
# Pick the size of the overlap region to be 127 bytes: tiktoken cl100k-base has
#   max token size 128byte. (remember that special unicode chars are 4byte...)
# This choice gives us a guarantee as long as the max vocab el. length does not
#   exceed 128 bytes: if a merge crosses the overlap boundary in one of two
#   overlapped chunks, it does NOT cross the overlap boundary in the other
#   overlapped chunk!
# So: for each chunk, we track if a merge occurs across its overlap boundary.
#   If it does, we mark that chunk (eg add a flag to the chunk data structure:
#   could just be a list of 2-el lists, 1 per chunk, which marks start/end
#   boundary crossing).
# After processing chunks, we de-overlap using this list. De-overlapping is easy if
#   only one of two elements is marked, or neither (we just preserve the overlapped region
#   from one chunk, or the first chunk in the latter case). If both elements are marked,
#   it means we merged across both overlap boundaries independently (because we set the
#   overlap region appropriately). We need to carefully stitch in this case.
# The above discussion applies to encoding. For training, we don't need to
#   worry about resolving across overlap boundaries, but we do need to avoid
#   double-counting...
@dataclass
class BPEImproved:
    """BPE-based utf8 tokenizer. With some optimizations"""

    corpus_path: Path = Path(".")
    max_vocab_size: int = 1024
    pretokenizer: Callable = pretokenizer_gpt2
    special_tokens: List[str] = field(default_factory=list)
    vocab: Dict[int, bytes] = field(default_factory=dict)
    merges: list[tuple[bytes, bytes]] = field(default_factory=list)
    vocab_inverse: Dict[bytes, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if len(self.vocab) == 0:
            # We allow for vocab/merges to be passed in from a pretrained tokenizer.
            self._map_special_tokens()
            self._train(self.corpus_path)
        # BUG: could be a bug, not being careful...
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}

    def _map_special_tokens(self):
        # Scheme: split the string on special tokens. Tokenize these manually.
        # Then operate normally on the rest. (no cou nting on special tokens)
        # NOTE: Assume "valid" special tokens (no substring issues)
        for i, token in enumerate(self.special_tokens):
            self.vocab[i] = token.encode("utf8")

    def _split_on_special_tokens(self, text: str, training=False) -> List[str]:
        """training=True strips the special tokens. False preserves them (for encoding)."""

        if len(self.special_tokens) == 0:
            # short circuit: empty regex breaks everything up
            return [text]
        patterns = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pat = "|".join(map(re.escape, patterns))
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
        count_dict_pretoken: Dict[Tuple[bytes, ...], int] = {}
        # count_dict_pair = Dict[tuple[int, int], int] = {}
        for pretoken in text:
            token = tuple(bytes([b]) for b in pretoken.encode("utf8"))
            if token not in count_dict_pretoken:
                count_dict_pretoken[token] = 1
            else:
                count_dict_pretoken[token] += 1
        while len(self.vocab) < self.max_vocab_size:
            # count frequencies
            count_dict: Dict[tuple[bytes, bytes], int] = {}
            for pretoken in count_dict_pretoken.keys():
                for pair in zip(pretoken, pretoken[1:]):
                    if pair not in count_dict:
                        count_dict[pair] = count_dict_pretoken[pretoken]
                    else:
                        count_dict[pair] += count_dict_pretoken[pretoken]
            # Most frequent scan
            most_frequent_pair = None
            max_count = -1
            for pair in sorted(
                count_dict.keys()
            ):  # Guarantees lexicographic merge ordering
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
            self.vocab[len(self.vocab)] = reduce(operator.add, most_frequent_pair)
            # Update the pretoken count dict (manual merge)
            # PERF: conversion to/from tuple is wasteful
            for pretoken in list(count_dict_pretoken):
                new_key = list(pretoken)  # can't avoid this?
                new_key = BPEImproved._merge(new_key, most_frequent_pair)
                new_key = tuple(new_key)  # keys need to be mutable...
                if new_key != pretoken:
                    count_dict_pretoken[new_key] = count_dict_pretoken[pretoken]
                    del count_dict_pretoken[pretoken]

    @staticmethod
    def _merge(token_seq: List[bytes], merge_pair: tuple[bytes, bytes]) -> List[bytes]:
        ptr = 0
        while ptr < len(token_seq) - 1:
            byte_seq = token_seq[ptr]
            next_byte_seq = token_seq[ptr + 1]
            if (byte_seq, next_byte_seq) == merge_pair:
                token_seq[ptr] = byte_seq + next_byte_seq
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
                segment_encoded = segment.encode("utf8")
                token = self.vocab_inverse[segment_encoded]
                text_encoded.append(token)
            else:
                segment_pretokenized = self.pretokenizer(segment)
                for segment in segment_pretokenized:
                    segment_encoded = list(bytes([b]) for b in segment.encode("utf8"))
                    for merge in self.merges:
                        segment_encoded = BPEImproved._merge(segment_encoded, merge)
                    tokens = [
                        self.vocab_inverse[byte_seq] for byte_seq in segment_encoded
                    ]
                    text_encoded += tokens
        return text_encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, tokens: list[int]) -> str:
        # The decoding process simply involves the vocab lookup.
        output = bytes([])
        for token in tokens:
            output += self.vocab[token]
        return output.decode("utf8", errors="replace")


def test_BPE_naive():
    corpus_path = Path("./test_data/test.txt")
    vocab_size = 512  # 'initial' size is 256 (bytes)
    tokenizer = BPENaive(corpus_path, vocab_size, special_tokens=["<|STOP|>"])

    test_str = (
        "Hello, world! This is a test.<|STOP|>여러분들, 안녕하세요? 12,34 1 -- 3 #$@$)@"
    ) * 1

    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_str


if __name__ == "__main__":
    # Some benchmarking
    pass
