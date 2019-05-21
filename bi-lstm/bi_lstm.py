# encoding=utf8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from data_utils import *


class BiLSTMModel(object):
    def __init__(
            self, vocab_size, num_tags, embedding_size, hidden_size, learning_rate=1e-3, idx_to_tag=None):

        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name="sequence_length")
        self.idx_to_tag = idx_to_tag

        def lstm_cell():
            cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
            return rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("bi-lstm"):
            #self.sequence_length = tf.cast(tf.reduce_sum(tf.sign(self.input_x)), tf.int32)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell(),
                                                                        lstm_cell(),
                                                                        self.embedded_inputs,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)

        with tf.name_scope("output"):
            output = tf.concat([output_fw, output_bw], axis=-1)
            seq_length = tf.shape(output)[1]
            output = tf.reshape(output, [-1, hidden_size * 2])

        with tf.name_scope("optimize"):
            #w = tf.Variable(tf.truncated_normal([hidden_size * 2, num_tags]), name="w")
            #b = tf.constant(0.1, shape=[num_tags], name="b")

            w = tf.get_variable("w", dtype=tf.float32, shape=[hidden_size * 2, num_tags])
            b = tf.get_variable("b", dtype=tf.float32, shape=[num_tags], initializer=tf.zeros_initializer())

            self.scores = tf.nn.xw_plus_b(output, w, b, name="score")
            self.predictions = tf.reshape(self.scores, [-1, seq_length, num_tags])
            log_likehood, self.transition_params = crf.crf_log_likelihood(self.predictions,
                                                                          self.input_y, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likehood)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(self.loss)
            #grads_and_vars = optimizer.compute_gradients(self.loss)
            #optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def predict_batch(self, words, sess):
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.)

        viterbi_sequences = []
        predictions, trans_params = sess.run(
            [self.predictions, self.transition_params], feed_dict=fd)

        for pred, sequence_length in zip(predictions, sequence_lengths):
            pred = pred[:sequence_length]
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                pred, trans_params)
            viterbi_sequences.append(viterbi_seq)

        return viterbi_sequences, sequence_lengths

    def predict(self, words_raw, sess):
        words = [get_processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)

        pred_ids, _ = self.predict_batch([words], sess)
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        word_ids, sequence_length = pad_sequence(words, 0)

        feed = {
            self.sequence_length: sequence_length,
            self.input_x: word_ids,
        }

        if labels is not None:
            feed[self.input_y] = labels

        if dropout is not None:
            feed[self.dropout_keep_prob] = dropout

        return feed, sequence_length