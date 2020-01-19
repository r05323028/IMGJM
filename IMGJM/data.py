# Copyright (c) 2019 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import re
import random
from abc import ABCMeta
from pathlib import Path
from typing import List, Tuple, Dict
from itertools import zip_longest
import fasttext
from more_itertools import flatten
import numpy as np
import tensorflow as tf
from IMGJM.utils import pad_char_sequences


class BaseDataset(metaclass=ABCMeta):
    def __init__(self,
                 data_dir: str = None,
                 resource: str = None,
                 char2id: Dict = None,
                 word2id: Dict = None,
                 shuffle: bool = True,
                 *args,
                 **kwargs):
        self.base_data_dir = Path('dataset')
        if resource:
            self.train_data_fp = self.base_data_dir / f'{data_dir}_{resource}' / 'train.txt'
            self.test_data_fp = self.base_data_dir / f'{data_dir}_{resource}' / 'test.txt'
        else:
            self.train_data_fp = self.base_data_dir / data_dir / 'train.txt'
            self.test_data_fp = self.base_data_dir / data_dir / 'test.txt'
        with open(self.train_data_fp, 'r') as train_file:
            self.raw_train_data = train_file.readlines()
        with open(self.test_data_fp, 'r') as test_file:
            self.raw_test_data = test_file.readlines()
        self.char2id = char2id
        self.word2id = word2id
        if shuffle:
            random.shuffle(self.raw_train_data)
            random.shuffle(self.raw_test_data)

    @property
    def word2id(self):
        return self._word2id

    @word2id.setter
    def word2id(self, w2i):
        if w2i:
            self._word2id = w2i
        else:
            self._word2id = self.build_word2id(self.raw_train_data +
                                               self.raw_test_data)

    @property
    def pad_char_id(self) -> int:
        return len(self.char2id)

    @property
    def pad_word_id(self) -> int:
        return len(self.word2id)

    @property
    def char2id(self):
        return self._char2id

    @char2id.setter
    def char2id(self, c2i):
        if c2i:
            self._char2id = c2i
        else:
            self._char2id = self.build_char2id(self.raw_train_data +
                                               self.raw_test_data)

    @staticmethod
    def build_word2id(sentence_list: List) -> Dict:
        word2id = {}
        sentences = [
            re.sub(r'[\/p\/0\/n\n]', '', sent).split(' ')
            for sent in sentence_list
        ]
        sentences_flatten = sorted(list(set(flatten(sentences))))
        for i, word in enumerate(sentences_flatten):
            word2id[word] = i
        return word2id

    @staticmethod
    def build_char2id(sentence_list: List) -> Dict:
        char2id = {}
        sentences = [
            re.sub(r'[\/p\/0\/n\n]', '', sent) for sent in sentence_list
        ]
        char_list = sorted(list(set(''.join(sentences))))
        for i, char in enumerate(char_list):
            char2id[char] = i
        return char2id

    @staticmethod
    def parse_sentence(sentence: List) -> Tuple[List]:
        sent_, entity_, polarity_ = [], [], []
        e_count, p_count = 0, 0
        for word in sentence:
            if '/p' in word:
                sent_.append(word.replace('/p', ''))
                if e_count == 0:
                    entity_.append(1)
                    e_count += 1
                else:
                    entity_.append(2)
                if p_count == 0:
                    polarity_.append(1)
                    p_count += 1
                else:
                    polarity_.append(2)
            elif '/0' in word:
                sent_.append(word.replace('/0', ''))
                if e_count == 0:
                    entity_.append(1)
                    e_count += 1
                else:
                    entity_.append(2)
                if p_count == 0:
                    polarity_.append(3)
                    p_count += 1
                else:
                    polarity_.append(4)
            elif '/n' in word:
                sent_.append(word.replace('/n', ''))
                if e_count == 0:
                    entity_.append(1)
                    e_count += 1
                else:
                    entity_.append(2)
                if p_count == 0:
                    polarity_.append(5)
                    p_count += 1
                else:
                    polarity_.append(6)
            else:
                sent_.append(word)
                entity_.append(0)
                polarity_.append(0)
        return sent_, entity_, polarity_

    @staticmethod
    def format_dataset(sentence_list: List[str]) -> Tuple:
        sentences, entities, polarities = [], [], []
        data = [sent.replace('\n', '').split(' ') for sent in sentence_list]
        for sent in data:
            sent_, entity_, polarity_ = BaseDataset.parse_sentence(sent)
            sentences.append(sent_)
            entities.append(entity_)
            polarities.append(polarity_)
        return sentences, entities, polarities

    def transform(
            self,
            sentence_list: List[List]) -> Tuple[List[List[List]], List[List]]:
        '''
        Transform segmented sentence into char & word ids

        Args:
            sentence (list)

        Returns:
            char_ids (list)
            word_ids (list)
        '''
        char_ids, word_ids = [], []
        for sent in sentence_list:
            wids, cids = [], []
            for word in sent:
                wids.append(self.word2id.get(word, len(self.word2id) + 1))
                cid_in_cid = []
                for char in word:
                    cid_in_cid.append(
                        self.char2id.get(char,
                                         len(self.char2id) + 1))
                cids.append(cid_in_cid)
            word_ids.append(wids)
            char_ids.append(cids)
        return char_ids, word_ids

    def merge_all(self, sentences: List[str]) -> Tuple[List]:
        sents, entities, polarities = self.format_dataset(sentences)
        char_ids, word_ids = self.transform(sents)
        sequence_length = [len(sent) for sent in word_ids]
        return char_ids, word_ids, sequence_length, entities, polarities

    def merge_and_pad_all(self, sentences: List[str]) -> Tuple[np.ndarray]:
        sents, entities, polarities = self.format_dataset(sentences)
        char_ids, word_ids = self.transform(sents)
        pad_char_ids = pad_char_sequences(char_ids,
                                          padding='post',
                                          value=len(self.char2id))
        pad_word_ids = tf.keras.preprocessing.sequence.pad_sequences(
            word_ids, padding='post', value=len(self.word2id))
        sequence_length = [len(sent) for sent in word_ids]
        pad_entities = tf.keras.preprocessing.sequence.pad_sequences(
            entities, padding='post', value=0)
        pad_polarities = tf.keras.preprocessing.sequence.pad_sequences(
            polarities, padding='post', value=0)
        return pad_char_ids, pad_word_ids, sequence_length, pad_entities, pad_polarities

    @property
    def train_data(self) -> Tuple[List]:
        outputs = self.merge_all(self.raw_train_data)
        return outputs

    @property
    def test_data(self) -> Tuple[np.ndarray]:
        outputs = self.merge_all(self.raw_test_data)
        return outputs

    @property
    def pad_train_data(self) -> Tuple[np.ndarray]:
        outputs = self.merge_and_pad_all(self.raw_train_data)
        return outputs

    @property
    def pad_test_data(self) -> Tuple[np.ndarray]:
        outputs = self.merge_and_pad_all(self.raw_test_data)
        return outputs

    def batch_generator(self,
                        batch_size: int = 32,
                        training: bool = True) -> Tuple[np.ndarray]:
        if training:
            dataset = self.raw_train_data
        else:
            dataset = self.raw_test_data
        start, end = 0, batch_size
        batch_nums = (len(dataset) // batch_size) + 1
        for _ in range(batch_nums):
            pad_char_ids, pad_word_batch, sequence_length, pad_entities, pad_polarities = self.merge_and_pad_all(
                dataset[start:end])
            yield (pad_char_ids, pad_word_batch, sequence_length, pad_entities,
                   pad_polarities)
            start, end = end, end + batch_size


class SemEval2014(BaseDataset):
    def __init__(self,
                 data_dir: str = '14semeval',
                 char2id: Dict = None,
                 word2id: Dict = None,
                 resource: str = 'laptop',
                 *args,
                 **kwargs):
        super(SemEval2014, self).__init__(data_dir=data_dir,
                                          resource=resource,
                                          char2id=char2id,
                                          word2id=word2id,
                                          *args,
                                          **kwargs)


class Twitter(BaseDataset):
    def __init__(self,
                 data_dir: str = 'Twitter',
                 char2id: Dict = None,
                 word2id: Dict = None,
                 *args,
                 **kwargs):
        super(Twitter, self).__init__(data_dir=data_dir,
                                      resource=None,
                                      char2id=char2id,
                                      word2id=word2id,
                                      *args,
                                      **kwargs)


class KKBOXSentimentData(BaseDataset):
    def __init__(self,
                 data_dir: str = 'kkbox',
                 char2id: Dict = None,
                 word2id: Dict = None,
                 fasttext_model_fp: str = 'embeddings/wiki.zh.bin',
                 *args,
                 **kwargs):
        super(KKBOXSentimentData, self).__init__(data_dir=data_dir,
                                                 char2id=char2id,
                                                 word2id=word2id,
                                                 *args,
                                                 **kwargs)
        self.fasttext_model = fasttext.load_model(fasttext_model_fp)

    def merge_and_pad_all(self, sentences: List[str]) -> Tuple[np.ndarray]:
        sents, entities, polarities = self.format_dataset(sentences)
        char_ids, word_ids = self.transform(sents)
        pad_char_ids = pad_char_sequences(char_ids,
                                          padding='post',
                                          value=len(self.char2id))
        sequence_length = [len(sent) for sent in word_ids]
        pad_word_vecs = self.word2vec(sents, max_sent_len=max(sequence_length))
        pad_entities = tf.keras.preprocessing.sequence.pad_sequences(
            entities, padding='post', value=0)
        pad_polarities = tf.keras.preprocessing.sequence.pad_sequences(
            polarities, padding='post', value=0)
        return pad_char_ids, pad_word_vecs, sequence_length, pad_entities, pad_polarities

    def word2vec(self, sentences: List[List], max_sent_len: int) -> np.ndarray:
        res = []
        for sent in sentences:
            vecs = []
            for _, word in zip_longest(range(max_sent_len), sent):
                if word:
                    vec = self.fasttext_model.get_word_vector(word)
                    vecs.append(vec)
                else:
                    vecs.append(
                        np.zeros(shape=[self.fasttext_model.get_dimension()]))
            res.append(vecs)
        return np.array(res)