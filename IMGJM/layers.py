# Copyright (c) 2020 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union, Tuple
import numpy as np
import tensorflow as tf


class GloveEmbedding(tf.keras.layers.Layer):
    def __init__(self, glove_embedding_matrix: np.ndarray):
        super(GloveEmbedding, self).__init__()
        self.glove_embedding_matrix = glove_embedding_matrix

    def build(self, input_shape: tf.TensorShape):
        self.embeddings = tf.constant(self.glove_embedding_matrix)

    def call(self, inputs: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        word_embeddings = tf.nn.embedding_lookup(self.embeddings, inputs)
        return tf.cast(word_embeddings, tf.float32)


class Gate(tf.keras.layers.Layer):
    '''
    Gate layer

    Attributes:
        dim_a (int): dimension of a.
        dim_b (int): dimension of b.
    '''
    def __init__(self, dim_a: int, dim_b: int):
        super(Gate, self).__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b

    def build(self, input_shape: tf.TensorShape):
        self.W_g = self.add_weight('W_g',
                                   shape=[self.dim_b, self.dim_b],
                                   dtype=tf.float32)
        self.W_trans = self.add_weight('W_trans',
                                       shape=[self.dim_b, self.dim_a],
                                       dtype=tf.float32)

    def call(self,
             a: tf.Tensor,
             b: tf.Tensor,
             training: bool = False) -> tf.Tensor:
        '''
        Args:
            a (tf.Tensor)
            b (tf.Tensor)
        
        Returns:
            output (tf.Tensor)
        '''
        g_ = tf.matmul(a, self.W_trans, transpose_b=True)
        g = tf.nn.sigmoid(tf.matmul(g_, self.W_g))
        output = g * tf.matmul(a, self.W_trans, transpose_b=True) + (1 - g) * b
        return output


class CharEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int = 50,
                 window_size: int = 3,
                 filter_nums: int = 50,
                 *args,
                 **kwargs):
        super(CharEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.filter_nums = filter_nums
        embedding_matrix_ = np.random.uniform(-0.1,
                                              0.1,
                                              size=(vocab_size,
                                                    embedding_size))
        pad = np.zeros(shape=(1, embedding_size))
        unk = np.random.uniform(-0.1, 0.1, size=(1, embedding_size))
        self.embedding_matrix = np.concatenate((embedding_matrix_, pad, unk),
                                               axis=0)

    def build(self, input_shape: tf.TensorShape):
        self.embedding_weights = self.add_weight(
            'embedding_weights',
            shape=self.embedding_matrix.shape,
            initializer=tf.keras.initializers.Constant(self.embedding_matrix))
        self.conv_weights = self.add_weight(
            'conv_weights',
            shape=[self.window_size, self.embedding_size, self.filter_nums],
            dtype=tf.float32)
        self.conv_bias = self.add_weight('conv_bias', shape=[self.filter_nums])

    def call(self,
             inputs: Union[np.ndarray, tf.Tensor],
             training: bool = False) -> tf.Tensor:
        char_embedding = tf.nn.embedding_lookup(self.embedding_weights, inputs)
        sp = tf.shape(char_embedding)
        char_embedding_ = tf.reshape(char_embedding,
                                     shape=[sp[0] * sp[1], sp[2], sp[3]])
        cnn_output = tf.nn.conv1d(char_embedding_,
                                  filters=self.conv_weights,
                                  stride=1,
                                  padding='VALID')
        cnn_output = tf.nn.relu(cnn_output + self.conv_bias)
        cnn_output = tf.reduce_max(cnn_output, axis=1)
        cnn_output = tf.reshape(cnn_output,
                                shape=[sp[0], sp[1], self.embedding_size])
        return cnn_output


class CoarseGrainedLayer(tf.keras.layers.Layer):
    '''
    Coarse-grained tagging layer
    d_h = 2(d_w + d_ch) = 700

    Attributes:
        hidden_nums (int): cell units nums
    '''
    def __init__(self, hidden_nums: int = 700):
        super(CoarseGrainedLayer, self).__init__()
        self.hidden_nums = hidden_nums

    def build(self, input_shape: tf.TensorShape):
        self.W_z = self.add_weight('W_z',
                                   shape=[2, self.hidden_nums],
                                   dtype=tf.float32)
        self.W_q = self.add_weight('W_q',
                                   shape=[2, self.hidden_nums],
                                   dtype=tf.float32)
        gru_fw = tf.keras.layers.GRU(units=int(self.hidden_nums / 2),
                                     return_sequences=True)
        gru_bw = tf.keras.layers.GRU(units=int(self.hidden_nums / 2),
                                     return_sequences=True,
                                     go_backwards=True)
        self.blstm = tf.keras.layers.Bidirectional(gru_fw,
                                                   backward_layer=gru_bw)

    def call(self,
             char_embedding: tf.Tensor,
             word_embedding: tf.Tensor,
             sequence_length: tf.Tensor,
             mask: tf.Tensor,
             training: bool = False) -> tf.Tensor:
        '''
        Args:
            char_embedding (tf.Tensor)
            word_embedding (tf.Tensor)
            sequence_length (tf.Tensor)
            mask (tf.Tensor)
            training (bool)

        Returns:
            z_T (tf.Tensor): coarse-grained target
            z_S (tf.Tensor): sentiment clue
            hidden_states (tf.Tensor): bi-GRU hidden states
        '''
        word_representation = tf.concat([char_embedding, word_embedding],
                                        axis=-1)
        hidden_states = self.blstm(word_representation,
                                   mask=mask,
                                   training=training)
        z_T = tf.nn.softmax(
            tf.matmul(hidden_states, self.W_z, transpose_b=True))
        z_S = tf.nn.softmax(
            tf.matmul(hidden_states, self.W_q, transpose_b=True))
        return (z_T, z_S, hidden_states)


class Interaction(tf.keras.layers.Layer):
    '''
    Interaction layer
    d_h = 2(d_w + d_ch) = 700

    Attributes:
        hidden_nums (int)
        C_tar (int)
        C_sent (int)
    '''
    def __init__(self,
                 hidden_nums: int = 700,
                 C_tar: int = 5,
                 C_sent: int = 7,
                 *args,
                 **kwargs):
        super(Interaction, self).__init__()
        self.hidden_nums = hidden_nums
        self.C_tar = C_tar
        self.C_sent = C_sent

    def build(self, input_shape: tf.TensorShape):
        self.W_att = self.add_weight(
            'W_att',
            shape=[self.hidden_nums, self.hidden_nums],
            dtype=tf.float32)
        self.W_l = self.add_weight('W_l',
                                   shape=[self.C_tar, 2],
                                   dtype=tf.float32)
        self.W_r = self.add_weight('W_r',
                                   shape=[self.C_sent, self.hidden_nums],
                                   dtype=tf.float32)
        self.gate_T = Gate(dim_a=2, dim_b=2)
        self.gate_S = Gate(dim_a=self.C_tar, dim_b=self.C_sent)

    def call(self,
             coarse_grained_target: tf.Tensor,
             sentiment_clue: tf.Tensor,
             hidden_states: tf.Tensor,
             training: bool = False) -> Tuple[tf.Tensor]:
        '''
        Args:
            coarse_grained_target (tf.Tensor)
            sentiment_clue (tf.Tensor)
            hidden_states (tf.Tensor)
            training (bool)
        
        Returns:
            l_T (tf.Tensor)
            l_S (tf.Tensor)
        '''
        # sentiment to target
        U = tf.nn.tanh(
            tf.matmul(tf.matmul(hidden_states, self.W_att),
                      hidden_states,
                      transpose_b=True))
        A = tf.nn.softmax(U)
        sa_T = tf.matmul(A, sentiment_clue)
        l_T = tf.matmul(self.gate_T(sentiment_clue, sa_T),
                        self.W_l,
                        transpose_b=True)
        # target to sentiment
        r = tf.matmul(A, hidden_states)
        att_S = tf.nn.relu(tf.matmul(r, self.W_r, transpose_b=True))
        l_S = self.gate_S(l_T, att_S)
        return (l_T, l_S)


class FineGrainedLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_nums: int = 700,
                 C_tar: int = 5,
                 C_sent: int = 7,
                 *args,
                 **kwargs):
        super(FineGrainedLayer, self).__init__()
        self.hidden_nums = hidden_nums
        self.C_tar = C_tar
        self.C_sent = C_sent

    def build(self, input_shape: tf.TensorShape):
        gru_fw = tf.keras.layers.GRU(units=int(self.hidden_nums / 2),
                                     return_sequences=True)
        gru_bw = tf.keras.layers.GRU(units=int(self.hidden_nums / 2),
                                     return_sequences=True,
                                     go_backwards=True)
        self.blstm = tf.keras.layers.Bidirectional(gru_fw,
                                                   backward_layer=gru_bw)
        self.W_ft = self.add_weight('W_ft',
                                    shape=[self.C_tar, self.hidden_nums],
                                    dtype=tf.float32)
        self.W_fs = self.add_weight('W_fs',
                                    shape=[self.C_sent, self.hidden_nums],
                                    dtype=tf.float32)
        self.gate_T = Gate(dim_a=self.C_tar, dim_b=self.C_tar)
        self.gate_S = Gate(dim_a=self.C_sent, dim_b=self.C_sent)

    def call(self,
             interacted_target: tf.Tensor,
             interacted_sentiment: tf.Tensor,
             hidden_states: tf.Tensor,
             sequence_length: tf.Tensor,
             mask: tf.Tensor,
             training: bool = False) -> tf.Tensor:
        hidden_states_ = self.blstm(hidden_states,
                                    mask=mask,
                                    training=training)
        f_T = tf.matmul(hidden_states_, self.W_ft, transpose_b=True)
        f_S = tf.matmul(hidden_states_, self.W_fs, transpose_b=True)
        o_T = self.gate_T(interacted_target, f_T)
        o_S = self.gate_S(interacted_sentiment, f_S)
        return (o_T, o_S)