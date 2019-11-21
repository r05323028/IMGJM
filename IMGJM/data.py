import re
from abc import ABCMeta
from pathlib import Path
from typing import List, Tuple, Dict
from more_itertools import flatten
import tensorflow as tf


class BaseDataset(metaclass=ABCMeta):
    def __init__(self,
                 data_dir: str = None,
                 resource: str = None,
                 char2id: Dict = None,
                 word2id: Dict = None,
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
        return sentences, entity_, polarities

    def transform(self, sentence_list: List[List]
                  ) -> Tuple[List[List[List]], List[List]]:
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
