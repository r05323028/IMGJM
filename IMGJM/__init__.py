# Copyright (c) 2019 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from abc import ABCMeta
from typing import Dict, List
import numpy as np
import tensorflow as tf
import tf_metrics
from IMGJM.layers import (CharEmbedding, GloveEmbedding, CoarseGrainedLayer,
                          Interaction, FineGrainedLayer)


class BaseModel(metaclass=ABCMeta):
    def build_tf_session(self,
                         logdir: str,
                         deploy: bool,
                         var_list: List[tf.Variable] = None):
        self.sess = tf.Session()
        if var_list:
            self.saver = tf.train.Saver(var_list=var_list)
        else:
            self.saver = tf.train.Saver()
        if not deploy:
            self.merged_all = tf.summary.merge_all()
            self.train_log_writer = tf.summary.FileWriter(
                logdir=logdir, graph=self.sess.graph)
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
        char_vocab_size (int)
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
                 char_vocab_size: int,
                 embedding_weights: np.ndarray,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 hidden_nums: int = 700,
                 dropout_rate: float = 0.5,
                 kernel_size: int = 3,
                 filter_nums: int = 50,
                 C_tar: int = 5,
                 C_sent: int = 7,
                 beta: float = 1.0,
                 gamma: float = 0.7,
                 logdir: str = 'logs',
                 deploy: bool = False,
                 *args,
                 **kwargs):
        # attributes
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = embedding_weights.shape[0]
        self.embedding_size = embedding_weights.shape[1]
        self.embedding_weights = embedding_weights
        self.embedding_shape = embedding_weights.shape
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

        # layers
        self.char_embedding = CharEmbedding(vocab_size=char_vocab_size)
        self.word_embedding = GloveEmbedding()
        self.coarse_grained_layer = CoarseGrainedLayer(hidden_nums=hidden_nums)
        self.interaction_layer = Interaction(hidden_nums=hidden_nums,
                                             C_tar=C_tar,
                                             C_sent=C_sent)
        self.fine_grained_layer = FineGrainedLayer(hidden_nums=hidden_nums,
                                                   C_tar=C_tar,
                                                   C_sent=C_sent)

        # build session
        self.build_model()
        var_list = [
            v for v in tf.global_variables()
            if v.name != 'Placeholders/WordEmbedding:0'
        ]
        self.build_tf_session(logdir=logdir, deploy=deploy, var_list=var_list)
        self.initialize_weights()
        self.initialize_embedding(embedding_weights)

    def initialize_embedding(self, embedding_weights: np.ndarray):
        with tf.name_scope('Initializing'):
            self.embedding_placeholder = tf.placeholder(
                tf.float32, shape=self.embedding_shape)
            self.sess.run(
                self.glove_embedding.assign(self.embedding_placeholder),
                feed_dict={self.embedding_placeholder: embedding_weights})

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
            self.glove_embedding = tf.Variable(tf.constant(
                0.0, shape=self.embedding_shape),
                                               trainable=False,
                                               name='WordEmbedding')
            self.training = tf.placeholder(dtype=tf.bool)
        with tf.name_scope('Hidden_layers'):
            char_embedding = self.char_embedding(self.char_ids)
            self.char_e = char_embedding
            word_embedding = self.word_embedding(
                self.word_ids, embedding_placeholder=self.glove_embedding)
            char_embedding = tf.layers.dropout(char_embedding,
                                               rate=self.dropout_rate,
                                               training=self.training)
            word_embedding = tf.layers.dropout(word_embedding,
                                               rate=self.dropout_rate,
                                               training=self.training)
            coarse_grained_target, self.sentiment_clue, hidden_states = self.coarse_grained_layer(
                char_embedding, word_embedding, self.sequence_length)
            interacted_target, interacted_sentiment = self.interaction_layer(
                coarse_grained_target, self.sentiment_clue, hidden_states)
            multi_grained_target, multi_grained_sentiment = self.fine_grained_layer(
                interacted_target, interacted_sentiment, hidden_states,
                self.sequence_length)
            multi_grained_target = tf.layers.dropout(multi_grained_target,
                                                     rate=self.dropout_rate,
                                                     training=self.training)
            multi_grained_sentiment = tf.layers.dropout(
                multi_grained_sentiment,
                rate=self.dropout_rate,
                training=self.training)
        with tf.name_scope('CRF'):
            with tf.name_scope('Target'), tf.variable_scope(
                    'Target_Variables', reuse=tf.AUTO_REUSE):
                target_log_likelihood, target_trans_params = tf.contrib.crf.crf_log_likelihood(
                    inputs=multi_grained_target,
                    tag_indices=self.y_target,
                    sequence_lengths=self.sequence_length)
            with tf.name_scope('Sentiment'), tf.variable_scope(
                    'Sentiment_Variables', reuse=tf.AUTO_REUSE):
                sentiment_log_likelihood, sentiment_trans_params = tf.contrib.crf.crf_log_likelihood(
                    inputs=multi_grained_sentiment,
                    tag_indices=self.y_sentiment,
                    sequence_lengths=self.sequence_length)
        with tf.name_scope('Loss'):
            loss_target = tf.reduce_mean(-target_log_likelihood)
            loss_sentiment = tf.reduce_mean(-sentiment_log_likelihood)
            loss_ol = tf.reduce_mean(coarse_grained_target[:, :, 0] *
                                     self.sentiment_clue[:, :, 0])
            # P(not O) = 1 - P(O)
            loss_brl = tf.losses.mean_squared_error(
                1 - tf.nn.softmax(multi_grained_target)[:, :, 0],
                1 - tf.nn.softmax(multi_grained_sentiment)[:, :, 0])
            self.total_loss = loss_target + loss_sentiment + self.beta * loss_ol + self.gamma * loss_brl
        with tf.name_scope('Optimization'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(
                self.total_loss,
                global_step=tf.train.get_or_create_global_step())
        with tf.name_scope('Prediction'):
            self.target_preds, _ = tf.contrib.crf.crf_decode(
                potentials=multi_grained_target,
                transition_params=target_trans_params,
                sequence_length=self.sequence_length)
            self.sentiment_preds, _ = tf.contrib.crf.crf_decode(
                potentials=multi_grained_sentiment,
                transition_params=sentiment_trans_params,
                sequence_length=self.sequence_length)
        with tf.name_scope('Metrics'):
            average = 'micro'
            with tf.name_scope('Train'):
                self.train_target_precision, self.train_target_precision_op = tf_metrics.precision(
                    self.y_target,
                    self.target_preds,
                    self.C_tar, [i for i in range(1, self.C_tar)],
                    average=average)
                self.train_target_recall, self.train_target_recall_op = tf_metrics.recall(
                    self.y_target,
                    self.target_preds,
                    self.C_tar, [i for i in range(1, self.C_tar)],
                    average=average)
                self.train_target_f1, self.train_target_f1_op = tf_metrics.f1(
                    self.y_target,
                    self.target_preds,
                    self.C_tar, [i for i in range(1, self.C_tar)],
                    average=average)
                self.train_sentiment_precision, self.train_sentiment_precision_op = tf_metrics.precision(
                    self.y_sentiment,
                    self.sentiment_preds,
                    self.C_sent, [i for i in range(1, self.C_sent)],
                    average=average)
                self.train_sentiment_recall, self.train_sentiment_recall_op = tf_metrics.recall(
                    self.y_sentiment,
                    self.sentiment_preds,
                    self.C_sent, [i for i in range(1, self.C_sent)],
                    average=average)
                self.train_sentiment_f1, self.train_sentiment_f1_op = tf_metrics.f1(
                    self.y_sentiment,
                    self.sentiment_preds,
                    self.C_sent, [i for i in range(1, self.C_sent)],
                    average=average)
            with tf.name_scope('Test'):
                self.test_target_precision, self.test_target_precision_op = tf_metrics.precision(
                    self.y_target,
                    self.target_preds,
                    self.C_tar, [i for i in range(1, self.C_tar)],
                    average=average)
                self.test_target_recall, self.test_target_recall_op = tf_metrics.recall(
                    self.y_target,
                    self.target_preds,
                    self.C_tar, [i for i in range(1, self.C_tar)],
                    average=average)
                self.test_target_f1, self.test_target_f1_op = tf_metrics.f1(
                    self.y_target,
                    self.target_preds,
                    self.C_tar, [i for i in range(1, self.C_tar)],
                    average=average)
                self.test_sentiment_precision, self.test_sentiment_precision_op = tf_metrics.precision(
                    self.y_sentiment,
                    self.sentiment_preds,
                    self.C_sent, [i for i in range(1, self.C_sent)],
                    average=average)
                self.test_sentiment_recall, self.test_sentiment_recall_op = tf_metrics.recall(
                    self.y_sentiment,
                    self.sentiment_preds,
                    self.C_sent, [i for i in range(1, self.C_sent)],
                    average=average)
                self.test_sentiment_f1, self.test_sentiment_f1_op = tf_metrics.f1(
                    self.y_sentiment,
                    self.sentiment_preds,
                    self.C_sent, [i for i in range(1, self.C_sent)],
                    average=average)

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
            self.training: True
        }
        metrics_ops = [
            self.train_target_precision_op, self.train_target_recall_op,
            self.train_target_f1_op, self.train_sentiment_precision_op,
            self.train_sentiment_recall_op, self.train_sentiment_f1_op
        ]
        metrics = [
            self.train_target_precision, self.train_target_recall,
            self.train_target_f1, self.train_sentiment_precision,
            self.train_sentiment_recall, self.train_sentiment_f1
        ]
        self.sess.run(self.train_op, feed_dict=feed_dict)
        self.sess.run(metrics_ops, feed_dict=feed_dict)
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
            self.training: False
        }
        metrics_ops = [
            self.test_target_precision_op, self.test_target_recall_op,
            self.test_target_f1_op, self.test_sentiment_precision_op,
            self.test_sentiment_recall_op, self.test_sentiment_f1_op
        ]
        metrics = [
            self.test_target_precision, self.test_target_recall,
            self.test_target_f1, self.test_sentiment_precision,
            self.test_sentiment_recall, self.test_sentiment_f1
        ]
        self.sess.run(metrics_ops, feed_dict=feed_dict)
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
            self.training: False
        }
        target_preds, sentiment_preds = self.sess.run(
            [self.target_preds, self.sentiment_preds], feed_dict=feed_dict)
        return target_preds, sentiment_preds

    def get_sentiment_clue(self, inputs: Dict):
        feed_dict = {
            self.char_ids: inputs.get('char_ids'),
            self.word_ids: inputs.get('word_ids'),
            self.sequence_length: inputs.get('sequence_length'),
            self.y_target: inputs.get('y_target'),
            self.y_sentiment: inputs.get('y_sentiment'),
            self.training: False
        }
        sentiment_clue = self.sess.run(self.sentiment_clue,
                                       feed_dict=feed_dict)
        return sentiment_clue
