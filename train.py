'''
Training script for IMGJM
'''
from typing import Dict, Tuple
from argparse import ArgumentParser
import logging
import coloredlogs
import numpy as np
from tqdm import tqdm, trange
from IMGJM import IMGJM
from IMGJM.data import (SemEval2014, Twitter)
from IMGJM.utils import build_glove_embedding, build_mock_embedding


class BoolParser:
    @classmethod
    def parse(cls, arg: str) -> bool:
        if arg.lower() in ['false', 'no']:
            return False
        else:
            return True


def get_logger(logger_name: str = 'IMGJM', level: str = 'INFO'):
    logger = logging.getLogger(logger_name)
    coloredlogs.install(
        level=level,
        fmt=
        f'%(asctime)s | %(name)-{len(logger_name) + 1}s| %(levelname)s | %(message)s',
        logger=logger)
    return logger


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=3)
    arg_parser.add_argument('--model_dir', type=str, default='outputs')
    arg_parser.add_argument('--mock_embedding',
                            type=BoolParser.parse,
                            default=False)
    return vars(arg_parser.parse_args())


def build_feed_dict(input_tuple: Tuple[np.ndarray]) -> Dict:
    pad_char_ids, pad_word_ids, sequence_length, pad_entities, pad_polarities = input_tuple
    feed_dict = {
        'char_ids': pad_char_ids,
        'word_ids': pad_word_ids,
        'sequence_length': sequence_length,
        'y_target': pad_entities,
        'y_sentiment': pad_polarities,
    }
    return feed_dict


def main(*args, **kwargs):
    np.random.seed(1234)
    logger = get_logger()
    if kwargs.get('mock_embedding'):
        logger.info('Initializing dataset...')
        dataset = SemEval2014()
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
        logger.info('Build mock embedding')
        _, embedding_weights = build_mock_embedding(dataset.word2id)
        logger.info('Building mock embedding finished')
    else:
        logger.info('Loading Glove embedding...')
        word2id, embedding_weights, _ = build_glove_embedding()
        logger.info('Embeding loaded.')
        logger.info('Initializing dataset...')
        dataset = SemEval2014(word2id=word2id)
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
    logger.info('Start training...')
    model = IMGJM(char_vocab_size=vocab_size,
                  embedding_weights=embedding_weights,
                  batch_size=kwargs.get('batch_size'))
    for _ in trange(kwargs.get('epochs'), desc='epoch'):
        train_batch_generator = tqdm(
            dataset.batch_generator(batch_size=kwargs.get('batch_size')),
            desc='training')
        for input_tuple in train_batch_generator:
            feed_dict = build_feed_dict(input_tuple)
            tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1 = model.train_on_batch(
                feed_dict)
            train_batch_generator.set_description(
                f'[Train][Target]: p-{tar_p:.3f}, r-{tar_r:.3f}, f1-{tar_f1:.3f} [Senti]: p-{sent_p:.3f}, r-{sent_r:.3f}, f1-{sent_f1:.3f}'
            )
        test_batch_generator = tqdm(dataset.batch_generator(
            batch_size=kwargs.get('batch_size'), training=False),
                                    desc='testing')
        for input_tuple in test_batch_generator:
            feed_dict = build_feed_dict(input_tuple)
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