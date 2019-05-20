#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import os

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


class CoNLLDataset(object):

    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """

        Args:
            filename:
            processing_word:
            processing_tag:
            max_iter:
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        n_iter = 0
        with open(self.filename, encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        n_iter += 1
                        if self.max_iter is not None and n_iter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """

    Args:
        datasets:

    Returns:

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {}".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocabs(dataset):
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    vocab = set()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            word = line.strip().split('  ')[0]
            vocab.update(word)

    return vocab


def write_vocab(vocab, filename):
    print("writing vocab...")
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    try:
        d = dict()
        with open(filename, encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except Exception as e:
        raise Exception

    return d


def export_trimmed_glove_vector(vocab, glove_filename, trimmed_filename, dim):
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('  ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]


def get_processing_word(vocab_words=None, vocab_chars=None,
                        lowercase=False, chars=False, allow_unk=True):
    def f(word):
        if vocab_chars is not None and chars is True:
            char_ids = []
            for char in word:
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that your vocab")

        if vocab_chars is not None and chars is True:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(max_length, len(seq)))

    return sequence_padded, sequence_length


def pad_sequence(sequences, pad_tok, nlevel=1):
    if nlevel == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevel == 2:
        max_length_word = max([max(map(lambda x: len(x)), seq) for seq in sequences])

        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded.append(sp)
            sequence_length.append(sl)

        max_sentence_length = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            max_length_word * [pad_tok], max_sentence_length)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_sentence_length)

    return sequence_padded, sequence_padded


def next_batch(data, batch_size):
    x_batch, y_batch = [], []

    for (x, y) in data:
        if len(x_batch) == batch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch.append(x)
        y_batch.append(y)

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):

    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[0]

    return tag_class, tag_type


def get_chunks(seq, tags):

    default = tags[NONE]
    idx_to_tag = {idx: tag for idx, tag in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif chunk_type != tok_chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq)-1)
        chunks.append(chunk)

    return chunks


def build_data(train_file, val_file, test_file, output_words, output_tags, output_chars, use_chars=False):

    # save raw vocab file
    processing_word = None
    train = CoNLLDataset(train_file, processing_word)
    val = CoNLLDataset(val_file, processing_word)
    test = CoNLLDataset(test_file, processing_word)

    vocab_words, vocab_tags = get_vocabs([train, val, test])

    vocab_words.add(UNK)
    vocab_words.add(NUM)

    write_vocab(vocab_words, output_words)
    write_vocab(vocab_tags, output_tags)

    vocab_chars = get_char_vocabs(train)
    write_vocab(vocab_chars, output_chars)

    # output the processed train file
    processing_word = get_processing_word(vocab_words, vocab_chars, lowercase=True, chars=use_chars)
    train = CoNLLDataset(train_file, processing_word)
    val = CoNLLDataset(val_file, processing_word)
    test = CoNLLDataset(test_file, processing_word)

    return train, val, test, vocab_words, vocab_tags


