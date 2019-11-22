# Copyright (c) 2019 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from abc import ABCMeta
from typing import Dict
import numpy as np
import tensorflow as tf
from IMGJM.layers import (CharEmbedding, GloveEmbedding, CoarseGrainedLayer,
                          Interaction, FineGrainedLayer)


class BaseModel(metaclass=ABCMeta):
    def build_tf_session(self, logdir: str):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.merged_all = tf.summary.merge_all()
        self.train_log_writer = tf.summary.FileWriter(logdir=logdir,
                                                      graph=self.sess.graph)
        self.test_log_writer = tf.summary.FileWriter(logdir=logdir,
                                                     graph=self.sess.graph)

    def initialize_weights(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def save_model(self, model_dir: str):
        self.saver.save(sess=self.sess, save_path=model_dir)

    def load_model(self, model_dir: str):
        self.saver.restore(sess=self.sess, save_path=model_dir)


class IMGJM(BaseModel):
    '''
    Interactive Multi-Grained Joint Model for Targeted Sentiment Analysis (CIKM 2019)

    Attributes:
        vocab_size (int)
        embedding_weights (np.ndarray)
        batch_size (int) 
        learning_rate (float) 
        embedding_size (int) 
        hidden_nums (int) 
        dropout_rate (float) 
        kernel_size (int) 
        filter_nums (int) 
        C_tar (int) 
        C_sent (int) 
        beta (float) 
        gamma (float) 
    '''
    def __init__(self,
                 vocab_size: int,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 embedding_size: int = 300,
                 hidden_nums: int = 700,
                 dropout_rate: float = 0.5,
                 kernel_size: int = 3,
                 filter_nums: int = 50,
                 C_tar: int = 5,
                 C_sent: int = 7,
                 beta: float = 1.0,
                 gamma: float = 0.7,
                 logdir: str = 'logs',
                 model_dir: str = 'models',
                 *args,
                 **kwargs):
        # attributes
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.filter_nums = filter_nums
        self.C_tar = C_tar
        self.C_sent = C_sent
        self.beta = beta
        self.gamma = gamma
        self.logdir = logdir
        self.model_dir = model_dir

        # layers
        self.char_embedding = CharEmbedding(vocab_size=vocab_size)
        self.word_embedding = GloveEmbedding()
        self.coarse_grained_layer = CoarseGrainedLayer(hidden_nums=hidden_nums)
        self.interaction_layer = Interaction(hidden_nums=hidden_nums,
                                             C_tar=C_tar,
                                             C_sent=C_sent)
        self.fine_grained_layer = FineGrainedLayer(hidden_nums=hidden_nums,
                                                   C_tar=C_tar,
                                                   C_sent=C_sent)

        # build session
        self.build_tf_session(logdir=logdir)
        self.initialize_weights()

    def build_model(self):
        '''
        Model building function
        '''
        with tf.name_scope('Placeholders'):
            self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.char_ids = tf.placeholder(dtype=tf.int32,
                                           shape=[None, None, None])
            self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])
            self.y_target = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.y_sentiment = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None])
            self.glove_embedding = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, None])
            self.training = tf.placeholder(dtype=tf.bool)
        with tf.name_scope('Hidden_layers'):
            char_embedding = self.char_embedding(self.char_ids)
            word_embedding = self.word_embedding(
                self.word_ids, embedding_placeholder=self.glove_embedding)
            coarse_grained_target, sentiment_clue, hidden_states = self.coarse_grained_layer(
                char_embedding, word_embedding, self.sequence_length)
            interacted_target, interacted_sentiment = self.interaction_layer(
                coarse_grained_target, sentiment_clue, hidden_states)
            multi_grained_target, multi_grained_sentiment = self.fine_grained_layer(
                interacted_target, interacted_sentiment, self.sequence_length)
        with tf.name_scope('CRF'):
            target_log_likelihood, target_trans_params = tf.contrib.crf.crf_log_likelihood(
                inputs=multi_grained_target,
                tag_indices=self.y_target,
                sequence_lengths=self.sequence_length)
            sentiment_log_likelihood, sentiment_trans_params = tf.contrib.crf.crf_log_likelihood(
                inputs=multi_grained_sentiment,
                tag_indices=self.y_sentiment,
                sequence_lengths=self.sequence_length)
        with tf.name_scope('Loss'):
            loss_target = tf.reduce_mean(-target_log_likelihood)
            loss_sentiment = tf.reduce_mean(-sentiment_log_likelihood)
            loss_ol = tf.reduce_mean(coarse_grained_target[:, :, 0] *
                                     sentiment_log_likelihood[:, :, 0])
            loss_brl = tf.losses.mean_squared_error(
                tf.nn.softmax(multi_grained_target),
                tf.nn.softmax(multi_grained_sentiment))
            self.total_loss = loss_target + loss_sentiment + self.beta * loss_ol + self.gamma * loss_brl
        with tf.name_scope('Optimization'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(
                self.total_loss,
                global_step=tf.train.get_or_create_global_step())
        with tf.name_scope('Prediction'):
            self.target_preds = tf.contrib.crf.crf_decode(
                potentials=multi_grained_target,
                transition_params=target_trans_params,
                sequence_lengths=self.sequence_length)
            self.sentiment_preds = tf.contrib.crf.crf_decode(
                potentials=multi_grained_sentiment,
                transition_params=sentiment_trans_params,
                sequence_lengths=self.sequence_length)
        with tf.name_scope('Metrics'):
            self.target_precision, self.target_precision_op = tf.metrics.precision(
                self.y_target, self.target_preds)
            self.target_recall, self.target_recall_op = tf.metrics.recall(
                self.y_target, self.target_preds)
            self.target_f1, self.target_f1_op = tf.contrib.metrics.f1_score(
                self.y_target, self.target_preds)
            self.sentiment_precision, self.sentiment_precision_op = tf.metrics.precision(
                self.y_sentiment, self.sentiment_preds)
            self.sentiment_recall, self.sentiment_recall_op = tf.metrics.recall(
                self.y_sentiment, self.sentiment_preds)
            self.sentiment_f1, self.sentiment_f1_op = tf.contrib.metrics.f1_score(
                self.y_sentiment, self.sentiment_preds)

    def train_on_batch(self, inputs: Dict):
        '''
        Train function of IMGJM

        Args:
            inputs (dict)

        Returns:
            tar_p (float) 
            tar_r (float) 
            tar_f1 (float) 
            sent_p (float) 
            sent_r (float) 
            sent_f1 (float) 
        '''
        feed_dict = {
            self.char_ids: inputs.get('char_ids'),
            self.word_ids: inputs.get('word_ids'),
            self.sequence_length: inputs.get('sequence_length'),
            self.y_target: inputs.get('y_target'),
            self.y_sentiment: inputs.get('y_sentiment'),
            self.glove_embedding: inputs.get('glove_embedding'),
            self.training: True
        }
        ops = [
            self.train_op, self.target_precision_op, self.target_recall_op,
            self.sentiment_precision_op, self.sentiment_recall_op,
            self.sentiment_f1_op
        ]
        metrics = [
            self.target_precision, self.target_recall, self.target_f1,
            self.sentiment_precision, self.sentiment_recall, self.sentiment_f1
        ]
        self.sess.run(ops, feed_dict=feed_dict)
        tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1 = self.sess.run(metrics)
        return tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1

    def test_on_batch(self, inputs: Dict):
        '''
        Test function of IMGJM

        Args:
            inputs (dict)

        Returns:
            tar_p (float) 
            tar_r (float) 
            tar_f1 (float) 
            sent_p (float) 
            sent_r (float) 
            sent_f1 (float) 
        '''
        feed_dict = {
            self.char_ids: inputs.get('char_ids'),
            self.word_ids: inputs.get('word_ids'),
            self.sequence_length: inputs.get('sequence_length'),
            self.y_target: inputs.get('y_target'),
            self.y_sentiment: inputs.get('y_sentiment'),
            self.glove_embedding: inputs.get('glove_embedding'),
            self.training: False
        }
        ops = [
            self.target_precision_op, self.target_recall_op,
            self.sentiment_precision_op, self.sentiment_recall_op,
            self.sentiment_f1_op
        ]
        metrics = [
            self.target_precision, self.target_recall, self.target_f1,
            self.sentiment_precision, self.sentiment_recall, self.sentiment_f1
        ]
        self.sess.run(ops, feed_dict=feed_dict)
        tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1 = self.sess.run(metrics)
        return tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1

    def predict_on_batch(self, inputs: Dict):
        '''
        Predict function of IMGJM

        Args:
            inputs (dict)

        Returns:
            target_preds (np.ndarray)
            sentiment_preds (np.ndarray)
        '''
        feed_dict = {
            self.char_ids: inputs.get('char_ids'),
            self.word_ids: inputs.get('word_ids'),
            self.sequence_length: inputs.get('sequence_length'),
            self.y_target: inputs.get('y_target'),
            self.y_sentiment: inputs.get('y_sentiment'),
            self.glove_embedding: inputs.get('glove_embedding'),
            self.training: False
        }
        target_preds, sentiment_preds = self.sess.run(
            [self.target_preds, self.sentiment_preds], feed_dict=feed_dict)
        return target_preds, sentiment_preds