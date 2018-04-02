import pickle
import numpy as np
import torch
import random
import os

class AGE2(object):
    def __init__(self, datapath, batch_size=50):
        self.batch_size = batch_size
        self.datapath = datapath
        
        data_file = open(self.datapath, 'rb')
        self.train_set, self.dev_set, self.test_set = pickle.load(data_file)
        self.weight = torch.FloatTensor(pickle.load(data_file).astype('float32'))

        self._wtoi = pickle.load(data_file)
        self.wtoi = self._wtoi.get # dict.get
        self.itow = pickle.load(data_file).__getitem__ # list.__getitem__

        self.id_to_tf = lambda _: 0
        self.num_words = len(self._wtoi)
        data_file.close()

        self.train_size = len(self.train_set)
        self.dev_size = len(self.dev_set)
        self.test_size = len(self.test_set)
        self.train_ptr = 0
        self.dev_ptr = 0
        self.test_ptr = 0

    def wrap_numpy_to_longtensor(self, *args):
        return map(torch.LongTensor, args) 


    def train_minibatch_generator(self):
        while self.train_ptr <= self.train_size - self.batch_size:
            self.train_ptr += self.batch_size
            minibatch = self.train_set[self.train_ptr - self.batch_size : self.train_ptr]
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
            self.train_ptr = 0
            random.shuffle(self.train_set)
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
    data_dir = '/data/sjx/dataset/self-attentive-age2/'
    save_path = './data/age2.pickle'

    print("loading GloVe...")
    w1 = {}
    for line in open(glove_path):
        line=line.split(' ')
        w1[line[0]] = np.asarray([float(x) for x in line[1:]]).astype('float32')

    f1 = os.path.join(data_dir, 'age2_train')
    f2 = os.path.join(data_dir, 'age2_valid')
    f3 = os.path.join(data_dir, 'age2_test')
    # note that class No. = rating -1
    classname = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
    f = [f1, f2, f3]

    print("processing dataset, 3 dots to punch: ")
    w2 = {}
    w2['<pad>'] = np.zeros((1, len(w1['the'])), dtype='float32')
    wtoi = {'<pad>': 0}  # reserve 0 for future padding
    itow = ['<pad>']
    vocab_count = 1
    train_dev_test = []
    for file in f:
        pairs = []
        for line in open(file):
            line=line.strip().split()
            s1 = line[1:]
            s1[0]=s1[0].lower()

            rate_score = classname[line[0]]

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
        train_dev_test.append(pairs)

    print("augmenting word embedding vocabulary...")
    mean_words = np.zeros((len(w1['the']),))
    mean_words_std = 1e-1

    npy_rng = np.random.RandomState(123)
    for k in w2:
        if len(w2[k]) != 0:
            w2[k] = sum(w2[k]) / len(w2[k])  # mean of all surounding words
        else:
            w2[k] = mean_words + npy_rng.randn(mean_words.shape[0]) * \
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
