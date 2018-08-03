'''
Created on July 20, 2018
@author : hsiaoyetgun (yqxiao)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Model import ESIM
import os
from Utils import *
import sys
from datetime import datetime
import Config

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# feed data into feed_dict
def feed_data(premise, premise_mask, hypothesis, hypothesis_mask, y_batch,
              dropout_keep_prob):
    feed_dict = {model.premise: premise,
                 model.premise_mask: premise_mask,
                 model.hypothesis: hypothesis,
                 model.hypothesis_mask: hypothesis_mask,
                 model.y: y_batch,
                 model.dropout_keep_prob: dropout_keep_prob}
    return feed_dict

# evaluate current model on devset
def evaluate(sess, premise, premise_mask, hypothesis, hypothesis_mask, y):
    batches = next_batch(premise, premise_mask, hypothesis, hypothesis_mask, y)
    data_nums = len(premise)
    total_loss = 0.0
    total_acc = 0.0
    for batch in batches:
        batch_nums = len(batch[0])
        feed_dict = feed_data(*batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_nums
        total_acc += acc * batch_nums
    return total_loss / data_nums, total_acc / data_nums

# training
def train():
    # load data
    print_log('Loading training and validation data ...', file=log)
    start_time = time.time()
    premise_train, premise_mask_train, hypothesis_train, hypothesis_mask_train, y_train = sentence2Index(arg.trainset_path, vocab_dict)
    premise_dev, premise_mask_dev, hypothesis_dev, hypothesis_mask_dev, y_dev = sentence2Index(arg.devset_path, vocab_dict)
    print(len(premise_train), len(premise_dev))
    data_nums = len(premise_train)
    time_diff = get_time_diff(start_time)
    print_log('Time usage : ', time_diff, file=log)

    # model saving
    saver = tf.train.Saver(max_to_keep=5)
    save_file_dir, save_file_name = os.path.split(arg.save_path)
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)

    # for TensorBoard
    print_log('Configuring TensorBoard and Saver ...', file=log)
    if not os.path.exists(arg.tfboard_path):
        os.makedirs(arg.tfboard_path)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(arg.tfboard_path)

    # init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(), {model.embed_matrix : embeddings})

    # count trainable parameters
    total_parameters = count_parameters()
    print_log('Total trainable parameters : {}'.format(total_parameters), file=log)

    # training
    print_log('Start training and evaluating ...', file=log)
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved_batch = 0
    isEarlyStop = False
    for epoch in range(arg.num_epochs):
        print_log('Epoch : ', epoch + 1, file=log)
        batches = next_batch(premise_train, premise_mask_train, hypothesis_train, hypothesis_mask_train, y_train, batchSize=arg.batch_size)
        total_loss, total_acc = 0.0, 0.0
        for batch in batches:
            batch_nums = len(batch[0])
            feed_dict = feed_data(*batch, arg.dropout_keep_prob)
            _, batch_loss, batch_acc = sess.run([model.train, model.loss, model.acc], feed_dict=feed_dict)
            total_loss += batch_loss * batch_nums
            total_acc += batch_acc * batch_nums

            # evaluta on devset
            if total_batch % arg.eval_batch == 0:
                # write tensorboard scalar
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

                feed_dict[model.dropout_keep_prob] = 1.0
                loss_val, acc_val = evaluate(sess, premise_dev, premise_mask_dev, hypothesis_dev, hypothesis_mask_dev, y_dev)

                # save model
                saver.save(sess = sess, save_path = arg.save_path + '_dev_loss_{:.4f}.ckpt'.format(loss_val))
                # save best model
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved_batch = total_batch
                    saver.save(sess = sess, save_path = arg.best_path)
                    improved_flag = '*'
                else:
                    improved_flag = ''

                # show batch training information
                time_diff = get_time_diff(start_time)
                msg = 'Epoch : {0:>3}, Batch : {1:>8}, Train Batch Loss : {2:>6.2}, Train Batch Acc : {3:>6.2%}, Dev Loss : {4:>6.2}, Dev Acc : {5:>6.2%}, Time : {6} {7}'
                print_log(msg.format(epoch + 1, total_batch, batch_loss, batch_acc, loss_val, acc_val, time_diff, improved_flag))

            total_batch += 1
            # early stop judge
            if total_batch - last_improved_batch > arg.early_stop_step:
                print_log('No optimization for a long time, auto-stopping ...', file = log)
                isEarlyStop = True
                break
        if isEarlyStop:
            break

        time_diff = get_time_diff(start_time)
        total_loss, total_acc = total_loss / data_nums, total_acc / data_nums
        msg = '** Epoch : {0:>2} finished, Train Loss : {1:>6.2}, Train Acc : {2:6.2%}, Time : {3}'
        print_log(msg.format(epoch + 1, total_loss, total_acc, time_diff), file = log)

if __name__ == '__main__':
    # read config
    config = Config.ModelConfig()
    arg = config.arg

    vocab_dict = load_vocab(arg.vocab_path)
    arg.vocab_dict_size = len(vocab_dict)

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
    print_log('CMD : python3 {0}'.format(' '.join(sys.argv)), file = log)
    print_log('Training with following options :', file = log)
    print_args(arg, log)

    model = ESIM(arg.seq_length, arg.n_vocab, arg.embedding_size, arg.hidden_size, arg.attention_size, arg.n_classes,\
                 arg.batch_size, arg.learning_rate, arg.optimizer, arg.l2, arg.clip_value)
    train()
    log.close()