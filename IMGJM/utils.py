# Copyright (c) 2019 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from typing import Dict, Tuple, List
from itertools import zip_longest
from more_itertools import flatten
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def transform_glove_to_gensim_format(
        input_file: str = 'embeddings/glove.840B.300d.txt',
        output_file: str = 'embeddings/glove.840B.300d.word2vec.txt',
        keep_origin_data: bool = False):
    glove2word2vec(input_file, output_file)
    if not keep_origin_data:
        os.system(f'rm {input_file}')


def build_glove_embedding(
        embedding_path: str = 'embeddings/glove.840B.300d.word2vec.txt'
) -> Tuple[Dict, np.ndarray, KeyedVectors]:
    '''
    PAD_INDEX: 2196016
    UNK_INDEX: 2196017
    '''
    glove_model = KeyedVectors.load_word2vec_format(embedding_path,
                                                    binary=False)
    word2id = {word: vocab.index for word, vocab in glove_model.vocab.items()}
    embeddings = glove_model.vectors
    pad = np.zeros(shape=(1, embeddings.shape[1]))
    unk = np.random.uniform(-0.25, 0.25, size=(1, embeddings.shape[1]))
    embeddings = np.concatenate((embeddings, pad, unk), axis=0)
    return word2id, embeddings, glove_model


def build_mock_embedding(word2id: Dict,
                         embedding_size: int = 300) -> Tuple[Dict, np.ndarray]:
    vocab_size = len(word2id)
    embedding = np.random.uniform(-0.25,
                                  0.25,
                                  size=(vocab_size, embedding_size))
    pad = np.zeros(shape=(1, embedding_size))
    unk = np.random.uniform(-0.25, 0.25, size=(1, embedding_size))
    embedding = np.concatenate((embedding, pad, unk), axis=0)
    return word2id, embedding


def pad_char_sequences(sequences: List, padding='post',
                       value=0.0) -> np.ndarray:
    max_sent_len = max([len(sent) for sent in sequences])
    max_word_len = max(
        [max([len(word) for word in sent]) for sent in sequences])
    sequences_ = []
    for sent in sequences:
        sent_ = tf.keras.preprocessing.sequence.pad_sequences(
            sent, maxlen=max_word_len, padding=padding, value=value)
        if len(sent_) < max_sent_len:
            if padding == 'post':
                sent_ = np.concatenate(
                    (sent_,
                     np.full(shape=(max_sent_len - len(sent_), max_word_len),
                             fill_value=value)),
                    axis=0)
            else:
                sent_ = np.concatenate(
                    (np.full(shape=(max_sent_len - len(sent_), max_word_len),
                             fill_value=value), sent_),
                    axis=0)
        sequences_.append(sent_)
    return np.array(sequences_)