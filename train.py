# Copyright (c) 2019 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
'''
Training script for IMGJM
'''
from typing import Dict, Tuple
from argparse import ArgumentParser
import logging
import yaml
import coloredlogs
import numpy as np
from tqdm import tqdm, trange
from IMGJM import IMGJM
from IMGJM.data import (SemEval2014, Twitter, KKBOXSentimentData)
from IMGJM.utils import build_glove_embedding, build_mock_embedding


class BoolParser:
    @classmethod
    def parse(cls, arg: str) -> bool:
        if arg.lower() in ['false', 'no']:
            return False
        else:
            return True


def get_logger(logger_name: str = 'IMGJM',
               level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger(logger_name)
    coloredlogs.install(
        level=level,
        fmt=
        f'%(asctime)s | %(name)-{len(logger_name) + 1}s| %(levelname)s | %(message)s',
        logger=logger)
    return logger


def get_args() -> Dict:
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=50)
    arg_parser.add_argument('--model_dir', type=str, default='outputs')
    arg_parser.add_argument('--model_config_fp',
                            type=str,
                            default='model_settings.yml')
    arg_parser.add_argument('--embedding', type=str, default='glove')
    arg_parser.add_argument('--dataset', type=str, default='laptop')
    return vars(arg_parser.parse_args())


def build_feed_dict(input_tuple: Tuple[np.ndarray],
                    input_type: str = 'ids') -> Dict:
    if input_type == 'ids':
        pad_char_ids, pad_word_ids, sequence_length, pad_entities, pad_polarities = input_tuple
        feed_dict = {
            'char_ids': pad_char_ids,
            'word_ids': pad_word_ids,
            'sequence_length': sequence_length,
            'y_target': pad_entities,
            'y_sentiment': pad_polarities,
        }
        return feed_dict
    else:
        pad_char_ids, pad_word_embedding, sequence_length, pad_entities, pad_polarities = input_tuple
        feed_dict = {
            'char_ids': pad_char_ids,
            'word_embedding': pad_word_embedding,
            'sequence_length': sequence_length,
            'y_target': pad_entities,
            'y_sentiment': pad_polarities,
        }
        return feed_dict


def load_model_config(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return config


def main(*args, **kwargs):
    np.random.seed(1234)
    logger = get_logger()
    if kwargs.get('embedding') == 'mock':
        logger.info('Initializing dataset...')
        if kwargs.get('dataset') == 'laptop':
            dataset = SemEval2014(resource='laptop')
        elif kwargs.get('dataset') == 'rest':
            dataset = SemEval2014(resource='rest')
        elif kwargs.get('dataset') == 'kkbox':
            dataset = KKBOXSentimentData()
        else:
            dataset = Twitter()
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
        logger.info('Build mock embedding')
        _, embedding_weights = build_mock_embedding(dataset.word2id)
        logger.info('Building mock embedding finished')
    elif kwargs.get('embedding') == 'glove':
        logger.info('Loading Glove embedding...')
        word2id, embedding_weights, _ = build_glove_embedding()
        logger.info('Embeding loaded.')
        logger.info('Initializing dataset...')
        if kwargs.get('dataset') == 'laptop':
            dataset = SemEval2014(word2id=word2id, resource='laptop')
        elif kwargs.get('dataset') == 'rest':
            dataset = SemEval2014(word2id=word2id, resource='rest')
        elif kwargs.get('dataset') == 'kkbox':
            dataset = KKBOXSentimentData(word2id=word2id)
        else:
            dataset = Twitter(word2id=word2id)
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
    elif kwargs.get('embedding') == 'fasttext':
        logger.info('Initializing dataset...')
        if kwargs.get('dataset') == 'laptop':
            dataset = SemEval2014(resource='laptop')
        elif kwargs.get('dataset') == 'rest':
            dataset = SemEval2014(resource='rest')
        elif kwargs.get('dataset') == 'kkbox':
            dataset = KKBOXSentimentData()
        else:
            dataset = Twitter()
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
    else:
        logger.warning('Invalid embedding choice.')
    logger.info('Loading model...')
    config = load_model_config(kwargs.get('model_config_fp'))
    if kwargs.get('embedding') == 'fasttext':
        model = IMGJM(char_vocab_size=vocab_size,
                      embedding_size=dataset.fasttext_model.get_dimension(),
                      input_type='embedding',
                      dropout=False,
                      **config['custom'])
    else:
        model = IMGJM(char_vocab_size=vocab_size,
                      embedding_weights=embedding_weights,
                      dropout=False,
                      **config['custom'])
    logger.info('Model loaded.')
    logger.info('Start training...')
    for _ in trange(kwargs.get('epochs'), desc='epoch'):
        train_batch_generator = tqdm(
            dataset.batch_generator(batch_size=kwargs.get('batch_size')),
            desc='training')
        for input_tuple in train_batch_generator:
            if kwargs.get('embedding') == 'fasttext':
                feed_dict = build_feed_dict(input_tuple,
                                            input_type='embedding')
            else:
                feed_dict = build_feed_dict(input_tuple, input_type='ids')
            tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1 = model.train_on_batch(
                feed_dict)
            train_batch_generator.set_description(
                f'[Train][Target]: p-{tar_p:.3f}, r-{tar_r:.3f}, f1-{tar_f1:.3f} [Senti]: p-{sent_p:.3f}, r-{sent_r:.3f}, f1-{sent_f1:.3f}'
            )
        test_batch_generator = tqdm(dataset.batch_generator(
            batch_size=kwargs.get('batch_size'), training=False),
                                    desc='testing')
        for input_tuple in test_batch_generator:
            if kwargs.get('embedding') == 'fasttext':
                feed_dict = build_feed_dict(input_tuple,
                                            input_type='embedding')
            else:
                feed_dict = build_feed_dict(input_tuple, input_type='ids')
            tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1 = model.test_on_batch(
                feed_dict)
            test_batch_generator.set_description(
                f'[Test][Target]: p-{tar_p:.3f}, r-{tar_r:.3f}, f1-{tar_f1:.3f} [Senti]: p-{sent_p:.3f}, r-{sent_r:.3f}, f1-{sent_f1:.3f}'
            )
        model.save_model(kwargs.get('model_dir') + '/' + 'model')
    logger.info('Training finished.')


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)