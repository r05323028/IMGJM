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
    np.random.seed(1234)
    logger = get_logger()
    # logger.info('Loading Glove embedding...')
    # word2id, embedding_weights, _ = build_glove_embedding()
    # logger.info('Embeding loaded.')
    logger.info('Initializing dataset...')
    dataset = SemEval2014()
    vocab_size = len(dataset.char2id)
    logger.info('Dataset loaded.')
    logger.info('Build mock embedding')
    w2i, embedding_weights = build_mock_embedding(dataset.word2id)
    logger.info('Building mock embedding finished')
    model = IMGJM(vocab_size=vocab_size, batch_size=kwargs.get('batch_size'))
    s = ['did not enjoy the new windows 8 and touchscreen functions .']
    model.load_model('outputs' + '/' + 'model')
    inputs = dataset.merge_and_pad_all(s)
    feed_dict = build_feed_dict(inputs, embedding_weights)
    print([sent.replace('\n', '').split(' ') for sent in s])
    print(model.predict_on_batch(feed_dict)[0])
    print(model.predict_on_batch(feed_dict)[1])
    print(model.get_sentiment_clue(feed_dict)[:, :, 1])


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)