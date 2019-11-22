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
from IMGJM.utils import build_glove_embedding


def get_logger(logger_name: str = 'IMGJM', level: str = 'INFO'):
    logger = logging.getLogger(logger_name)
    coloredlogs.install(
        level=level,
        fmt='%(asctime)s | %(name)-6s| %(levelname)s | %(message)s',
        logger=logger)
    return logger


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=3)
    arg_parser.add_argument('--model_dir', type=str, default='outputs')
    return vars(arg_parser.parse_args())


def build_feed_dict(input_tuple: Tuple[np.ndarray],
                    embedding_weights: np.ndarray) -> Dict:
    pad_char_ids, pad_word_ids, sequence_length, pad_entities, pad_polarities = input_tuple
    feed_dict = {
        'char_ids': pad_char_ids,
        'word_ids': pad_word_ids,
        'sequence_length': sequence_length,
        'y_target': pad_entities,
        'y_sentiment': pad_polarities,
        'glove_embedding': embedding_weights,
    }
    return feed_dict


def main(*args, **kwargs):
    logger = get_logger()
    logger.info('Loading Glove embedding...')
    word2id, embedding_weights, _ = build_glove_embedding()
    logger.info('Embeding loaded.')
    logger.info('Initializing dataset...')
    dataset = SemEval2014(word2id=word2id)
    vocab_size = len(dataset.char2id)
    logger.info('Dataset loaded.')
    logger.info('Start training...')
    model = IMGJM(vocab_size=vocab_size, batch_size=kwargs.get('batch_size'))
    for _ in trange(kwargs.get('epochs'), desc='epoch'):
        batch_generator = tqdm(
            dataset.batch_generator(batch_size=kwargs.get('batch_size')),
            desc='training')
        for input_tuple in batch_generator:
            feed_dict = build_feed_dict(input_tuple, embedding_weights)
            tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1 = model.train_on_batch(
                feed_dict)
            batch_generator.set_description(
                f'[Target]: p-{tar_p}, r-{tar_r}, f1-{tar_f1} [Senti]: p-{sent_p}, r-{sent_r}, f1-{sent_f1}'
            )
        model.save_model(kwargs.get('model_dir'))
    logger.info('Training finished.')


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)