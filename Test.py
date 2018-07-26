'''
Created on July 20, 2018
@author : hsiaoyetgun (yqxiao)
Reference : Enhanced LSTM for Natural Language Inference (ACL 2017)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from Utils import *
import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import f1_score, roc_auc_score
from Model import ESIM
import Config

# testing
def predict():
    # load data
    print_log('Loading testing data ...', file=log)
    start_time = time.time()
    premise_test, premise_mask_test, hypothesis_test, hypothesis_mask_test, y_test = sentence2Index(
        arg.testset_path, vocab_dict)
    time_diff = get_time_diff(start_time)
    print_log('Time usage : ', time_diff, file=log)

    with tf.Session() as sess:
        # load model
        print("Loading model: " + arg.model)
        saver = tf.train.import_meta_graph(arg.model + '.meta')
        saver.restore(sess, arg.best_path)

        # testing
        print_log('Start testing ...', file=log)
        start_time = time.time()
        feed_dict = {model.premise: premise_test,
                     model.premise_mask: premise_mask_test,
                     model.hypothesis: hypothesis_test,
                     model.hypothesis_mask: hypothesis_mask_test,
                     model.dropout_keep_prob: 1.0}
        logits, loss, acc = sess.run([model.logits, model.loss, model.acc], feed_dict = feed_dict)

        # evaluating
        y_pred = np.argmax(logits, 1)
        y_true = np.argmax(y_test, 1)
        f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_true))
        auc = roc_auc_score(y_true, y_pred, average='weighted')
        for id in range(len(y_true)):
            print('Wrong classification:')
            if y_true[id] != y_pred[id]:
                premise_text = ''.join([index2word[idx] for idx in premise_test[id]])
                hypothesis_text = ''.join([index2word[idx] for idx in hypothesis_test[id]])
                print('Left_text: {0} / Right_text: {1}'.format(premise_text, hypothesis_text))
                print('The true label is {0} / The pred label is {1}'.format(y_true[id], y_pred[id]))
        print('The test accuracy / f1 / auc: {0[0]:.2%} / {0[1]:.4f} / {0[2]:.4f}'.format((acc, f1, auc)))
        time_diff = get_time_diff(start_time)
        print('Time usage: ', time_diff, '\n')

if __name__ == '__main__':
    # read config
    config = Config.ModelConfig()
    arg = config.arg

    vocab_dict = load_vocab(arg.vocab_path)
    arg.vocab_dict_size = len(vocab_dict)
    index2word = {index : word for word, index in vocab_dict}

    if arg.embedding_path:
        embeddings = load_embeddings(arg.embedding_path, vocab_dict)
    else:
        embeddings = init_embeddings(vocab_dict, arg.embedding_size)
    arg.n_vocab, arg.embedding_size = embeddings.shape

    if arg.embedding_normalize:
        embeddings = normalize_embeddings(embeddings)

    arg.n_classes = len(CATEGORIE_ID)

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    arg.log_path = 'config/log/log.{}'.format(dt)
    log = open(arg.log_path, 'w')
    print_log('CMD : python3 {0}'.format(' '.join(sys.argv)), file=log)
    print_log('Testing with following options :', file=log)
    print_args(arg, log)

    model = ESIM(arg.seq_length, arg.n_vocab, arg.embedding_size, arg.hidden_size, arg.attention_size, arg.n_classes, \
                 arg.batch_size, arg.learning_rate, arg.optimizer, arg.l2, arg.clip_value)
    predict()
    log.close()