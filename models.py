from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import init
from util import sequence_mask, unwrap_scalar_variable
import math
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
        self.dictionary = kwargs['dictionary']
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.attender:
            if type(layer) == nn.Linear:
                init.kaiming_normal(layer.weight.data)
                if layer.bias is not None:
                    init.constant(layer.bias.data, val=0)

    def forward(self, input, length, words, display):
        # input is [bsz, len, 2*nhid]
        # length is [bsz, ]
        # words is [bsz, len]
        bsz, l, nhid2 = input.size()
        mask = sequence_mask(length, max_length=l) # [bsz, len]
        compressed_embeddings = input.view(-1, nhid2)  # [bsz*len, 2*nhid]

        alphas = self.attender(compressed_embeddings)  # [bsz*len, hop]
        alphas = alphas.view(bsz, l, -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = self.softmax(alphas.view(-1, l))  # [bsz*hop, len]
        alphas = alphas.view(bsz, -1, l)  # [bsz, hop, len]

        mask = mask.unsqueeze(1).expand(-1, self.att_hops, -1) # [bsz, hop, len]
        alphas = alphas * mask.float() + 1e-20
        alphas = alphas / alphas.sum(2, keepdim=True) # renorm

        info = []
        if display:
            for i in range(bsz):
                s = '\n'
                for j in range(self.att_hops):
                    for k in range(unwrap_scalar_variable(length[i])):
                        s += '%s(%.2f) ' % (self.dictionary.itow(unwrap_scalar_variable(words[i][k])), unwrap_scalar_variable(alphas[i][j][k]))
                    s += '\n\n'
                info.append(s)

        supplements = {
                'attention': alphas, # [bsz, hop, len]
                'info': info,
                }

        return torch.bmm(alphas, input), supplements # [bsz, hop, 2*nhid]


class AvgBlockHieEncoder(nn.Module):

    def __init__(self, **kwargs):
        super(AvgBlockHieEncoder, self).__init__()
        # shared encoder
        self.attender = SelfAttentiveEncoder(**kwargs)
        self.att_hops = kwargs['att_hops']
        self.block_size = kwargs['block_size']
        self.dictionary = kwargs['dictionary']
        self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self, input, length, words, display): 
        # input is [bsz, len, 2*nhid]
        # length is [bsz, ]
        # words is [bsz, len]
        bsz, l, nhid2 = input.size()
        mask = sequence_mask(length, max_length=l) # [bsz, len]
        num_blocks = int(math.ceil(l / self.block_size))
        block_embeddings, block_alphas = [], []
        for i in range(num_blocks):
            begin = i * self.block_size
            end = min(l, self.block_size*(i+1))
            # e, a, info = self.attender(input[:, begin:end, :].contiguous(), mask[:, begin:end], display)
            e, a, _ = self.attender(input[:, begin:end, :].contiguous(), length-begin, words, display=False)
            block_embeddings.append(e)
            block_alphas.append(a)
        block_embeddings = torch.stack(block_embeddings) # [nblock, bsz, hop, 2*nhid]
        block_embeddings = block_embeddings.view(num_blocks, -1, nhid2) # [nblock, bsz*hop, 2*nhid]
        block_embeddings = torch.transpose(block_embeddings, 0, 1).contiguous() # [bsz*hop, nblock, 2*nhid]
        block_mask = mask[:, ::self.block_size].contiguous().unsqueeze(1).expand(-1, self.att_hops, -1).contiguous().view(-1, num_blocks) # [bsz*hop, nblock]
        # sent_embeddings, sent_alphas, sent_info = self.attender(block_embeddings, block_mask, display)
        sent_embeddings, sent_alphas, _ = self.attender(block_embeddings, block_mask.sum(dim=1), words, display=False)
        # [bsz*hop, hop, 2*nhid], [bsz*hop, hop, nblock]
        sent_embeddings = sent_embeddings.view(bsz, self.att_hops**2, nhid2) #[bsz, hop*hop, 2*nhid]

        # construct alphas
        # block_alphas is list of [bsz, hop, blocksz] of length nblock
        if num_blocks < self.att_hops: # TODO: hop should be 1 if num_blocks is small.
            sent_alphas = []
        else:
            sent_alphas = sent_alphas.view(bsz, self.att_hops, self.att_hops, num_blocks)
            sent_alphas = map(torch.squeeze, torch.chunk(torch.transpose(sent_alphas, 0, 1).contiguous(), chunks=self.att_hops)) # list of [bsz, hop, nblock] of length hop


        info = []
        if display:
            pass

        return sent_embeddings, block_alphas + sent_alphas, info


class HardSoftHieEncoder(nn.Module):

    def __init__(self, **kwargs):
        super(HardSoftHieEncoder, self).__init__()
        self.hard_attender = nn.Sequential(
                nn.Dropout(kwargs['dropout']),
                nn.Linear(kwargs['hidden_dim']*2, 300),
                nn.Tanh(),
                nn.Linear(300, 1),
                nn.Sigmoid(),
                ) # TODO other structures
        self.soft_attender = SelfAttentiveEncoder(**kwargs)
        self.att_hops = kwargs['att_hops']
        self.dictionary = kwargs['dictionary']
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.hard_attender:
            if type(layer) == nn.Linear:
                init.kaiming_normal(layer.weight.data)
                if layer.bias is not None:
                    init.constant(layer.bias.data, val=0)

    def forward(self, input, length, words, display): 
        # input is [bsz, len, 2*nhid]
        # mask is [bsz, len]
        # words is [bsz, len]
        bsz, l, nhid2 = input.size()
        mask = sequence_mask(length, max_length=l) # [bsz, len]
        z_soft = self.hard_attender(input.view(-1, nhid2)).view(bsz, l, 1) # [bsz, l, 1]
        z_hard = torch.bernoulli(z_soft).byte() & mask.unsqueeze(2)
        gate = (z_hard.float() - z_soft).detach() + z_soft
        gate_h = gate * input # [bsz, l, 2*nhid]
        '''
        # TODO: how to optimize the case when gate=0
        new_length = z_hard.int().sum(dim=1).squeeze().long() # [bsz, ]
        new_l = unwrap_scalar_variable(torch.max(new_length))
        new_input = [[Variable(input.data.new(nhid2).zero_())]*new_l for _ in range(bsz)]
        for i in range(bsz):
            k = 0
            for j in range(l): # TODO faster iteration
                if unwrap_scalar_variable(z_hard[i][j])==1:
                    new_input[i][k] = gate_h[i][j]
                    k += 1
            new_input[i] = torch.stack(new_input[i])
        new_input = torch.stack(new_input) # [bsz, new_l, 2*nhid]
        new_mask = sequence_mask(new_length, max_length=new_l) # [bsz, new_l]
        return self.soft_attender(new_input, new_mask)
        '''
        embeddings, alphas, _ = self.soft_attender(gate_h, length, words, False)

        info = []
        if display:
            for i in range(bsz):
                s = '\n'
                for j in range(self.att_hops):
                    for k in range(unwrap_scalar_variable(length[i])):
                        if unwrap_scalar_variable(z_hard[i][k]) == 1:
                            s += '%s(%.2f) ' % (self.dictionary.itow(unwrap_scalar_variable(words[i][k])), unwrap_scalar_variable(alphas[i][j][k]))
                        else:
                            s += '--%s-- ' % (self.dictionary.itow(unwrap_scalar_variable(words[i][k])))
                    s += '\n\n'
                info.append(s)

        supplements = {
                'attention': alphas,
                'info': info,
                }

        return embeddings, supplements


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

        #####################################################################
        if kwargs['model_type'] == 'sa':
            self.encoder = SelfAttentiveEncoder(**kwargs)
            self.encoder_out_dim = kwargs['hidden_dim'] * 2 * kwargs['att_hops']
        #####################################################################
        elif kwargs['model_type'] == 'avgblock':
            self.encoder = AvgBlockHieEncoder(**kwargs)
            self.encoder_out_dim = kwargs['hidden_dim']*2*kwargs['att_hops']*kwargs['att_hops']
        #####################################################################
        elif kwargs['model_type'] == 'hard':
            self.encoder = HardSoftHieEncoder(**kwargs)
            self.encoder_out_dim = kwargs['hidden_dim'] * 2 * kwargs['att_hops']
        #####################################################################
        else:
            raise Exception('unsupported type')
        #####################################################################
        self.tanh = nn.Tanh()
        self.classifier = nn.Sequential(
                nn.Dropout(kwargs['dropout']),
                nn.Linear(self.encoder_out_dim, kwargs['clf_hidden_dim']),
                nn.Tanh(),
                nn.Dropout(kwargs['dropout']),
                nn.Linear(kwargs['clf_hidden_dim'], kwargs['num_classes'])
                )
        self.dictionary = kwargs['dictionary']
        self.att_hops = kwargs['att_hops']
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


    def forward(self, words, length, display=False):
        words_t = words.t() # [len, bsz]
        words_embed = self.drop(self.word_embedding(words_t)) # [len, bsz, word_dim]
        words_h = self.bilstm(words_embed)[0] # [len, bsz, 2*nhid]
        words_h = torch.transpose(words_h, 0, 1).contiguous() # [bsz, len, 2*nhid]

        sent_h, supplements = self.encoder(words_h, length, words, display)
        sent_h = sent_h.view(sent_h.size(0), -1) # [bsz, ...]
        pred = self.classifier(sent_h)
        
        return pred, supplements

