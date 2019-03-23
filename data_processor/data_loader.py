#coding=utf-8
import fileinput
import numpy as np
import sys
import traceback
import tensorflow.contrib.keras as kr

from collections import Counter

class DataProcessor(object):
    def __init__(self, coding = 'utf-8'):
        self._coding = coding
        return

    def __read_file(self, file_name):
        contents, labels = [], []
        for line in fileinput.input(file_name):
            try:
                lable, content = line.strip().decode(self._coding).split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(lable)
            except:
                pass
        return contents, labels

    def build_vocob(self, train_path, vocob_path, vocob_size = 5000):
        data_train, _ = self.__read_file(train_path)
        all_data = []
        for content in data_train:
            all_data.extend(content)

        counter = Counter(all_data)
        count_pairs = counter.most_common(vocob_size - 1)
        words, _ = list(zip(*count_pairs))
        words = ['<PAD>'] + list(words)
        open(vocob_path, "wb").write('\n'.join(x.encode(self._coding) for x in words) + '\n')

    def read_category(self):
        categories = [u'体育', u'财经', u'房产', u'家居', u'教育', u'科技', u'时尚', u'时政', u'游戏', u'娱乐']
        cat2id = dict(zip(categories, range(len(categories))))
        return categories, cat2id

    def read_vocab(self, vocob_path):
        words = []
        for line in fileinput.input(vocob_path):
            try:
                line = line.strip().decode(self._coding)
                words.append(line)
            except:
                traceback.print_exc()
        words2id = dict(zip(words, range(len(words))))
        return words, words2id

    def process_file(self, filename, word2id, cat2id, max_length = 600):
        contents, labels = self.__read_file(filename)
        data_id, label_id = [], []
        for i in xrange(len(contents)):
            data_id.append([word2id[x] for x in contents[i] if x in word2id])
            label_id.append(cat2id[labels[i]])

        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat2id))
        return x_pad, y_pad

    def batch_iter(self, x, y, batch_size = 64):
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in xrange(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id : end_id], y_shuffle[start_id : end_id]
