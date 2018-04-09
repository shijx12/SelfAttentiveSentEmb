from __future__ import print_function
from __future__ import division
import pickle
import numpy as np
import torch
import random
import os
import codecs
import csv
from tqdm import tqdm
import time
import math
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from util import spacy_tokenize
random.seed(666)
np.random.seed(666)

class Yahoo(object):
    def __init__(self, datapath, batch_size=32):
        self.batch_size = batch_size
        self.datapath = datapath
        
        tic = time.time()
        data_file = open(self.datapath, 'rb')
        self.train_set, self.dev_set, self.test_set = pickle.load(data_file)
        self.weight = torch.FloatTensor(pickle.load(data_file).astype('float32'))

        self._wtoi = pickle.load(data_file) # NOTE: words are stored in unicode
        self.wtoi = self._wtoi.get # dict.get
        self.itow = pickle.load(data_file).__getitem__ # list.__getitem__

        self.id_to_tf = lambda _: 0
        self.num_words = len(self._wtoi)
        data_file.close()
        print('It takes %.2f sec to load datafile. train/dev/test: %d/%d/%d.' % (time.time() - tic, len(self.train_set), len(self.dev_set), len(self.test_set)))

        self.train_size = len(self.train_set)
        self.dev_size = len(self.dev_set)
        self.test_size = len(self.test_set)
        self.train_num_batch = int(math.ceil(self.train_size / self.batch_size))
        tic = time.time()
        self.trainset_bucket_shuffle()
        print('It takes %.2f sec to shuffle trainset.' % (time.time() - tic))
        self.train_ptr = 0
        self.dev_ptr = 0
        self.test_ptr = 0

    def wrap_numpy_to_longtensor(self, *args):
        return map(torch.LongTensor, args) 

    def trainset_bucket_shuffle(self):
        self.train_set.sort(key=lambda e: len(e[0])) # sort based on length
        shuffle_unit = 1000
        for i in range(0, self.train_size, shuffle_unit): # shuffle for every unit
            tmp = self.train_set[i: i+shuffle_unit]
            random.shuffle(tmp)
            self.train_set[i: i+shuffle_unit] = tmp
        self.train_iter_idx = list(range(0, self.train_num_batch))
        random.shuffle(self.train_iter_idx)

    def train_minibatch_generator(self):
        while self.train_ptr < self.train_num_batch:
            i = self.train_iter_idx[self.train_ptr]
            minibatch = self.train_set[i*self.batch_size : (i+1)*self.batch_size]
            l = np.max(map(lambda x: len(x[0]), minibatch), axis=0)
            sentences = np.zeros((self.batch_size, l), dtype='int32')
            labels = np.zeros((self.batch_size,), dtype='int32')
            lengths = np.zeros((self.batch_size,), dtype='int32')
            for i, (s, l) in enumerate(minibatch):
                sentences[i, :len(s)] = s
                labels[i] = l
                lengths[i] = len(s)
            self.train_ptr += 1
            yield self.wrap_numpy_to_longtensor(sentences, lengths, labels)
        else:
            self.train_ptr = 0
            self.trainset_bucket_shuffle()
            raise StopIteration


    # NOTE: for dev and test, all data should be fetched regardless of batch_size!
    def dev_minibatch_generator(self):
        while self.dev_ptr < self.dev_size:
            batch_size = min(self.batch_size, self.dev_size - self.dev_ptr)
            self.dev_ptr += batch_size
            minibatch = self.dev_set[self.dev_ptr - batch_size : self.dev_ptr]
            l = np.max(map(lambda x: len(x[0]), minibatch), axis=0)
            sentences = np.zeros((self.batch_size, l), dtype='int32')
            labels = np.zeros((self.batch_size,), dtype='int32')
            lengths = np.zeros((self.batch_size,), dtype='int32')
            for i, (s, l) in enumerate(minibatch):
                sentences[i, :len(s)] = s
                labels[i] = l
                lengths[i] = len(s)
            yield self.wrap_numpy_to_longtensor(sentences, lengths, labels)
        else:
            self.dev_ptr = 0
            raise StopIteration

    def test_minibatch_generator(self):
        while self.test_ptr < self.test_size:
            batch_size = min(self.batch_size, self.test_size - self.test_ptr)
            self.test_ptr += batch_size
            minibatch = self.test_set[self.test_ptr - batch_size : self.test_ptr]
            l = np.max(map(lambda x: len(x[0]), minibatch), axis=0)
            sentences = np.zeros((self.batch_size, l), dtype='int32')
            labels = np.zeros((self.batch_size,), dtype='int32')
            lengths = np.zeros((self.batch_size,), dtype='int32')
            for i, (s, l) in enumerate(minibatch):
                sentences[i, :len(s)] = s
                labels[i] = l
                lengths[i] = len(s)
            yield self.wrap_numpy_to_longtensor(sentences, lengths, labels)
        else:
            self.test_ptr = 0
            raise StopIteration


def dump():
    glove_path = '/data/share/glove.840B/glove.840B.300d.txt'
    data_dir = '/data/share/yahoo_answers_csv'
    save_path = '/data/sjx/self-attentive-Exp/data/yahoo.pickle'

    print("loading GloVe...")
    w1 = {}
    for line in open(glove_path):
        line=line.split(' ')
        w1[line[0]] = np.asarray([float(x) for x in line[1:]]).astype('float32')

    f1 = os.path.join(data_dir, 'train.csv')
    f2 = os.path.join(data_dir, 'test.csv')
    # note that class No. = rating -1
    classname = {str(i+1):i for i in range(10)}

    print("processing dataset")
    w2 = {}
    w2['<pad>'] = np.zeros((1, len(w1['the'])), dtype='float32')
    wtoi = {'<pad>': 0}  # reserve 0 for future padding
    itow = ['<pad>']
    vocab_count = 1
    train_dev_test = []

    print("train.csv...")
    pairs = []
    for line in tqdm(csv.reader(open(f1))):
        rate_score = classname[line[0]]
        s1 = spacy_tokenize(codecs.decode(' '.join(line[1:]), 'utf-8')) # concat all fields
        s1_words = []
        for word in s1:
            if word not in wtoi:
                wtoi[word] = vocab_count
                itow.append(word)
                vocab_count += 1
            s1_words.append(wtoi[word])
            if word not in w1:
                if word not in w2:
                    w2[word]=[]
                # find the WE for its surounding words
                for neighbor in s1:
                    if neighbor in w1:
                        w2[word].append(w1[neighbor])
        pairs.append((np.asarray(s1_words).astype('int32'),
                          rate_score))
    print(len(pairs))
    random.shuffle(pairs)
    num_dev = 60000
    print('split train/dev: %d/%d' % (len(pairs)-num_dev, num_dev))
    train_dev_test.append(pairs[:-num_dev])
    train_dev_test.append(pairs[-num_dev:])

    print("test.csv...") # 60,000
    pairs = []
    for line in tqdm(csv.reader(open(f2))):
        rate_score = classname[line[0]]
        s1 = spacy_tokenize(codecs.decode(' '.join(line[1:]), 'utf-8')) # concat all fields
        s1_words = []
        for word in s1:
            if word not in wtoi:
                wtoi[word] = vocab_count
                itow.append(word)
                vocab_count += 1
            s1_words.append(wtoi[word])
            if word not in w1:
                if word not in w2:
                    w2[word]=[]
                # find the WE for its surounding words
                for neighbor in s1:
                    if neighbor in w1:
                        w2[word].append(w1[neighbor])
        pairs.append((np.asarray(s1_words).astype('int32'),
                          rate_score))
    print(len(pairs))
    train_dev_test.append(pairs)

    print("augmenting word embedding vocabulary...")
    mean_words = np.zeros((len(w1['the']),))
    mean_words_std = 1e-1

    for k in w2:
        if len(w2[k]) != 0:
            w2[k] = sum(w2[k]) / len(w2[k])  # mean of all surounding words
        else:
            w2[k] = mean_words + np.random.randn(mean_words.shape[0]) * \
                                 mean_words_std * 0.1
    w2.update(w1)
    print("generating weight values...")

    ordered_word_embedding = [w2[word].reshape(1, -1) for word in itow]
    weight = np.concatenate(ordered_word_embedding, axis=0)

    print("dumping converted datasets...")
    save_file = open(save_path, 'wb')
    "the whole dataset, in list of list of tuples: list of train/valid/test set -> list of sentence pairs -> tuple with structure: (review, truth rate), all entries in numbers\n"
    pickle.dump(train_dev_test, save_file)
    "numpy.ndarray: a matrix with all referred words' embedding in its rows, embeddings are ordered by their corresponding word numbers.\n"
    pickle.dump(weight, save_file)
    "dict wtoi: word to their corresponding number\n"
    pickle.dump(wtoi, save_file)
    "list itow: number to words\n",
    pickle.dump(itow, save_file)
    save_file.close()


if __name__ == '__main__':
    dump()

