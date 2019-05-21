# encoding=utf8

import time
import datetime
import os
import numpy as np

import data_helpers
from bi_lstm import BiLSTMModel
from data_utils import *

import tensorflow as tf
from tensorflow.contrib import crf


# Data loading params
#tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("train_file", "./data/train_data.txt", "train_file")
tf.flags.DEFINE_string("val_file", "./data/val_data.txt", "val_file")
tf.flags.DEFINE_string("test_file", "./data/test_data.txt", "test_file")
tf.flags.DEFINE_string("checkpoint_path", "./model/", "checkpoint_path")

tf.flags.DEFINE_integer("early_stopping_max_epoch", 3, "early_stop_max_epoch")

tf.flags.DEFINE_string("words_file", "./data/words.txt", "words_file")
tf.flags.DEFINE_string("tags_file", "./data/tags.txt", "tags_file")
tf.flags.DEFINE_string("chars_file", "./data/chars.txt", "chars_file")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of rnn hidden layer (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def train(bi_lstm, train_data, val_data, vocab_words, vocab_tags):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            global_step = tf.Variable(0, name="global_step", trainable=False)
            start_time = time.time()

            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", start_time))
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            def train_step(train_data, val_data):
                best_score = 0
                no_imprv_epoch = 0

                for x_batch, y_batch in next_batch(train_data, FLAGS.batch_size):
                    cur_step = tf.train.global_step(sess, global_step)

                    fd = {
                        bi_lstm.input_x: x_batch,
                        bi_lstm.input_y: y_batch,
                        bi_lstm.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }

                    _, loss = sess.run([bi_lstm.train_op, bi_lstm.loss], feed_dict=fd)
                    print("step:{}, loss:{}".format(cur_step, loss))

                    if cur_step % FLAGS.evaluate_every == 0:
                        metrics = eval_step(val_data)
                        msg = " - ".join(["{} {:04.2f}".format(k, v)
                                          for k, v in metrics.items()])
                        print("eval result: " + msg)

                        score = metrics["f1"]
                        if score >= best_score:
                            no_imprv_epoch = 0
                            best_score = score
                            print("- new best score")
                            path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                            print("\nSave model checkpoint to {}\n".format(path))
                        else:
                            no_imprv_epoch += 1
                            if no_imprv_epoch >= FLAGS.early_stopping_max_epoch:
                                print("- early stopping {} epochs").format(no_imprv_epoch)
                                break

            def eval_step(val_data):
                accs = []
                correct_preds, total_correct, total_preds = 0., 0., 0.
                for words, labels in next_batch(val_data, FLAGS.batch_size):
                    labels_pred, sequence_length = bi_lstm.predict_batch(words, sess)

                    for lab, lab_pred, length in zip(labels, labels_pred, sequence_length):
                        lab = lab[:length]
                        lab_pred = lab_pred[:length]
                        accs += [a == b for (a, b) in zip(lab, lab_pred)]

                        lab_chunks = set(get_chunks(lab, vocab_tags))
                        lab_pred_chunks = set(get_chunks(lab_pred, vocab_tags))

                        correct_preds += len(lab_chunks & lab_pred_chunks)
                        total_preds += len(lab_pred_chunks)
                        total_correct += len(lab_chunks)
                p = correct_preds / total_preds if correct_preds > 0 else 0
                r = correct_preds / total_correct if correct_preds > 0 else 0
                f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
                acc = np.mean(accs)

                return {"acc": acc, "f1": f1}

            sess.run(tf.global_variables_initializer())
            train_step(train_data, val_data)


def main():
    train_data, val_data, test_data, vocab_words, vocab_tags = build_data(
        FLAGS.train_file, FLAGS.val_file, FLAGS.test_file, FLAGS.words_file, FLAGS.tags_file, FLAGS.chars_file)

    model = BiLSTMModel(vocab_size=len(vocab_words),
                        num_tags=len(vocab_tags),
                        embedding_size=FLAGS.embedding_size,
                        hidden_size=FLAGS.hidden_size)

    for d in train_data:
        print(d)
        break

    exit()

    train(model, train_data, test_data, vocab_words, vocab_tags)

if __name__ == '__main__':
    main()











