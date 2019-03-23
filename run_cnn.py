#coding=utf-8
import argparse
import datetime
import numpy as np
import os
import tensorflow as tf
import traceback

from sklearn import metrics

from model.cnn_model import TextCNNConfig, TextCNN
from data_processor.data_loader import DataProcessor

base_dir = 'data/'
train_path = os.path.join(base_dir, 'cnews.train.txt')
test_path = os.path.join(base_dir, 'cnews.test.txt')
val_path = os.path.join(base_dir, 'cnews.val.txt')
vocab_path = os.path.join(base_dir, 'cnews.vocab.txt')

saver_dir = 'checkpoints/textcnn/'
saver_path = os.path.join(saver_dir, 'checkpoints')

def get_time_diff(start_time):
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time).seconds
    return time_diff

def feed_data(x_batch, y_batch, keep_prob, model):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(data_processor, session, x_, y_, model):
    data_len = len(x_)
    batch_eval = data_processor.batch_iter(x_, y_, 64)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0, model)
        loss, acc = session.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len

def train(data_processor, word2id, cat2id, model, config):
    print ("Configuring Tensorboard and Saver...")
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        try:
            os.mkdir('tensorboard')
        except:
            traceback.print_exc()
            pass

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    if not os.path.exists(saver_dir):
        os.makedirs(saver_dir)

    print ("Loading train and validation data...")
    start_time = datetime.datetime.now()
    x_train, y_train = data_processor.process_file(train_path, word2id, cat2id, config.seq_length)
    x_val, y_val = data_processor.process_file(val_path, word2id, cat2id, config.seq_length)
    time_diff = get_time_diff(start_time)
    print ("[Time cost] Loading data time cost %d seconds" % time_diff)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print ("Traing and evaluating...")
    start_time = datetime.datetime.now()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in xrange(config.num_epochs):
        batch_train = data_processor.batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob, model)
            if total_batch % config.save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict = feed_dict)
                loss_val, acc_val = evaluate(data_processor, session, x_val, y_val, model)
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=saver_path)
                    improved_str = '*'
                else:
                    improved_str = ''
            time_diff = get_time_diff(start_time)
            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
            print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_diff, improved_str))
        session.run(model.optim, feed_dict = feed_dict)
        total_batch += 1

        if (total_batch - last_improved > require_improvement):
            print ("No optimization for a long time, auto-stopping...")
            falg = True
            break
        if flag:
            break

def test(data_processor, word2id, cat2id, model, config):
    print ("Loading test data...")
    start_time = datetime.datetime.now()
    x_test, y_test = data_processor.process_file(test_path, word2id, cat2id, config.seq_length)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess = session, save_path = saver_path)
    loss_test, acc_test = evaluate(data_processor, session, x_test, y_test, model)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))
    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1
    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape = len(x_test), dtype = np.int32)
    for i in xrange(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls))
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_diff(start_time)
    print("Time usage:", time_dif)

if __name__ == "__main__":
    data_processor = DataProcessor()
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', choices = ['train', 'test'])
    args = parser.parse_args()
    pattern = args.pattern
    if pattern not in ['train', 'test']:
        raise ValueError('Invalid pattern, must be train or test')
    config = TextCNNConfig()
    if not os.path.exists(vocab_path):
        data_processor.build_vocob(train_path, vocab_path, config.vocab_size)
    categories, cat2id = data_processor.read_category()
    words, word2id = data_processor.read_vocab(vocab_path)
    config.vocab_size = len(words)
    model = TextCNN(config)
    if pattern == 'train':
        train(data_processor, word2id, cat2id, model, config)
    else:
        test(data_processor, word2id, cat2id, model, config)
