# Copyright (c) 2019 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from typing import Dict, Tuple, List
from more_itertools import flatten
import numpy as np
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
