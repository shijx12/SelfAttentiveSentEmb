from __future__ import print_function
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import init
from util import sequence_mask
from IPython import embed


class SelfAttentiveEncoder(nn.Module):

    def __init__(self, **kwargs):
        super(SelfAttentiveEncoder, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.attender = nn.Sequential(
                nn.Dropout(kwargs['dropout']),
                nn.Linear(kwargs['hidden_dim']*2, kwargs['att_dim'], bias=False),
                nn.Tanh(),
                nn.Linear(kwargs['att_dim'], kwargs['att_hops'], bias=False),
                )
        self.att_hops = kwargs['att_hops']
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.attender:
            if type(layer) == nn.Linear:
                init.kaiming_normal(layer.weight.data)
                if layer.bias is not None:
                    init.constant(layer.bias.data, val=0)

    def forward(self, input, length):
        size = input.size()  # [bsz, len, 2*nhid]
        compressed_embeddings = input.view(-1, size[2])  # [bsz*len, 2*nhid]

        alphas = self.attender(compressed_embeddings)  # [bsz*len, hop]
        alphas = alphas.view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], -1, size[1])  # [bsz, hop, len]

        mask = sequence_mask(length, max_length=size[1]) # [bsz, len]
        mask = mask.unsqueeze(1).expand(-1, self.att_hops, -1) # [bsz, hop, len]
        alphas = alphas * mask.float() + 1e-20
        alphas = alphas / alphas.sum(2, keepdim=True) # renorm

        return torch.bmm(alphas, input), alphas # [bsz, hop, 2*nhid], [bsz, hop, len]


class Classifier(nn.Module):

    def __init__(self, **kwargs):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(kwargs['dropout'])
        self.word_embedding = nn.Embedding(num_embeddings=kwargs['num_words'], 
                                            embedding_dim=kwargs['word_dim'])
        self.bilstm = nn.LSTM(input_size=kwargs['word_dim'], 
                hidden_size=kwargs['hidden_dim'], 
                num_layers=kwargs['num_layers'], 
                dropout=kwargs['dropout'], 
                bidirectional=True)

        # if kwargs['type'] == 'sa':
        if True:
            self.encoder = SelfAttentiveEncoder(**kwargs)
            self.encoder_out_dim = kwargs['hidden_dim'] * 2 * kwargs['att_hops']
        else:
            raise Exception('unsupported type')
        self.tanh = nn.Tanh()
        self.classifier = nn.Sequential(
                nn.Dropout(kwargs['dropout']),
                nn.Linear(self.encoder_out_dim, kwargs['clf_hidden_dim']),
                nn.Tanh(),
                nn.Dropout(kwargs['dropout']),
                nn.Linear(kwargs['clf_hidden_dim'], kwargs['num_classes'])
                )
        self.dictionary = kwargs['dictionary']
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        for name, param in self.bilstm.named_parameters():
            if 'bias' in name:
                init.constant(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal(param)
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                init.kaiming_normal(layer.weight.data)
                if layer.bias is not None:
                    init.constant(layer.bias.data, val=0)


    def forward(self, words, length):
        words = words.t() # [len, bsz]
        words_embed = self.drop(self.word_embedding(words)) # [len, bsz, word_dim]
        words_h = self.bilstm(words_embed)[0] # [len, bsz, 2*nhid]
        words_h = torch.transpose(words_h, 0, 1).contiguous() # [bsz, len, 2*nhid]

        sent_h, attention = self.encoder(words_h, length)
        sent_h = sent_h.view(sent_h.size(0), -1) # [bsz, ...]
        pred = self.classifier(sent_h)
        supplements = {
                'attention': attention # [bsz, hop, len]
                }
        
        return pred, supplements

