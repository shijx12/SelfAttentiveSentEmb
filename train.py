from __future__ import print_function
from __future__ import division
from models import Classifier
from util import *
from dataset.age2 import AGE2

import torch
import tensorboard
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
import numpy as np
import time
import random
import argparse
import os
import logging
from IPython import embed



def train_iter(args, batch, **kw):
    model, params, criterion, optimizer = kw['model'], kw['params'], kw['criterion'], kw['optimizer']
    model.train()
    words, length, label = batch
    length = wrap_with_variable(length, volatile=False, cuda=args.cuda)
    words = wrap_with_variable(words, volatile=False, cuda=args.cuda)
    label = wrap_with_variable(label, volatile=False, cuda=args.cuda)
    logits, supplements = model.forward(words=words, length=length)
    label_pred = logits.max(1)[1]
    accuracy = torch.eq(label, label_pred).float().mean()
    num_correct = torch.eq(label, label_pred).long().sum()
    loss = criterion(input=logits, target=label)

    if args.penalization_coeff > 0:
        I = kw['I']
        attention = supplements['attention']
        attentionT = torch.transpose(attention, 1, 2).contiguous()
        extra_loss = Frobenius(torch.bmm(attention, attentionT) - I)
        loss += args.penalization_coeff * extra_loss

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(parameters=params, max_norm=0.5)
    optimizer.step()
    return loss, accuracy


def eval_iter(args, model, batch):
    model.eval()
    words, length, label = batch
    length = wrap_with_variable(length, volatile=False, cuda=args.cuda)
    words = wrap_with_variable(words, volatile=False, cuda=args.cuda)
    label = wrap_with_variable(label, volatile=False, cuda=args.cuda)
    logits, supplements = model(words=words, length=length)
    label_pred = logits.max(1)[1]
    num_correct = torch.eq(label, label_pred).long().sum()
    return num_correct, supplements 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-dim', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--hidden-dim', type=int, default=300, help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2, help='number of layers in BiLSTM')
    parser.add_argument('--att-dim', type=int, default=350, help='number of attention unit')
    parser.add_argument('--att-hops', type=int, default=4, help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--clf-hidden-dim', type=int, default=512, help='hidden (fully connected) layer size for classifier MLP')
    parser.add_argument('--penalization-coeff', type=float, default=1, help='the penalization coefficient')
    parser.add_argument('--clip', type=float, default=0.5, help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--lr', type=float, default=.001, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay rate per batch')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')


    parser.add_argument('--num-classes', type=int, default=5, help='number of classes')
    parser.add_argument('--data', type=str, default='./data/age2.pickle')
    parser.add_argument('--save-dir', type=str, required=True, help='path to save the final model')
    parser.add_argument('--fix-word-embedding', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    #######################################
    # a simple log file, the same content as stdout
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    ########################################
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))


    data = AGE2(datapath=args.data, batch_size=args.batch_size)

    model = Classifier(
        dictionary=data,
        dropout=args.dropout,
        num_words=data.num_words,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        word_dim=args.word_dim,
        att_dim=args.att_dim,
        att_hops=args.att_hops,
        clf_hidden_dim=args.clf_hidden_dim,
        num_classes=args.num_classes
    )
    logging.info(model)

    model.word_embedding.weight.data.set_(data.weight)
    if args.fix_word_embedding:
        model.word_embedding.weight.requires_grad = False
    if args.cuda:
        model = model.cuda()
    ''' count parameters
    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.word_embedding.weight.size())
    print('# of parameters: %d' % num_params)
    print('# of word embedding parameters: %d' % num_embedding_params)
    print('# of parameters (excluding word embeddings): %d' % (num_params - num_embedding_params))
    '''
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'sgd':
        optimizer_class = optim.SGD
    else:
        raise Exception('For other optimizers, please add it yourself. supported ones are: SGD and Adam.')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()
    # Identity matrix for each batch
    I = Variable(torch.eye(args.att_hops).unsqueeze(0).expand(args.batch_size, -1, -1))
    if args.cuda:
        I = I.cuda()
    trpack = {
            'model': model,
            'params': params, 
            'criterion': criterion, 
            'optimizer': optimizer,
            'I': I,
            }

    train_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'train'), flush_secs=10)
    valid_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'valid'), flush_secs=10)
    tsw, vsw = train_summary_writer, valid_summary_writer

    num_train_batches = data.train_size // data.batch_size 
    logging.info('num_train_batches: %d' % num_train_batches)
    validate_every = num_train_batches // 10
    best_vaild_accuacy = 0
    iter_count = 0
    tic = time.time()

    for epoch_num in range(args.max_epoch):
        for batch_iter, train_batch in enumerate(data.train_minibatch_generator()):
            progress = epoch_num + batch_iter / num_train_batches
            iter_count += 1

            train_loss, train_accuracy = train_iter(args, train_batch, **trpack)
            add_scalar_summary(tsw, 'loss', train_loss, iter_count)
            add_scalar_summary(tsw, 'acc', train_accuracy, iter_count)

            if (batch_iter + 1) % (num_train_batches // 100) == 0:
                tac = (time.time() - tic) / 60
                print('   %.2f minutes\tprogress: %.2f' % (tac, progress))
            if (batch_iter + 1) % validate_every == 0:
                correct_sum = 0
                for valid_batch in data.dev_minibatch_generator():
                    correct, supplements = eval_iter(args, model, valid_batch)
                    correct_sum += unwrap_scalar_variable(correct)
                valid_accuracy = correct_sum / data.dev_size 
                scheduler.step(valid_accuracy)
                add_scalar_summary(vsw, 'acc', valid_accuracy, iter_count)
                logging.info('Epoch %.2f: valid accuracy = %.4f' % (progress, valid_accuracy))
                if valid_accuracy > best_vaild_accuacy:
                    correct_sum = 0
                    for test_batch in data.test_minibatch_generator():
                        correct, supplements = eval_iter(args, model, test_batch)
                        correct_sum += unwrap_scalar_variable(correct)
                    test_accuracy = correct_sum / data.test_size
                    best_vaild_accuacy = valid_accuracy
                    model_filename = ('model-%.2f-%.3f-%.3f.pkl' % (progress, valid_accuracy, test_accuracy))
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print('Saved the new best model to %s' % model_path)



if __name__ == '__main__':
    main()

