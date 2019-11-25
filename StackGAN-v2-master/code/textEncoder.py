import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# import torchwordemb

class Opts():
    def __init__(self):
        self.embDim = 1024
        self.nRNNs = 1
        self.srnnDim = 1024
        self.irnnDim = 1024
        self.imfeatDim = 2014
        self.stDim = 1024
        self.ingrW2VDim = 1024
        self.instrW2VDim = 144
        self.maxSeqlen = 20

opts = Opts()

class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()
        
    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)


class instrRNN(nn.Module):
    def __init__(self):
        super(instrRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=opts.instrW2VDim, hidden_size=opts.srnnDim, bidirectional=True, batch_first=True)
        #_, vec = torchwordemb.load_word2vec_bin(opts.instrW2V)
        num_instr = 13
        self.embs = nn.Embedding(num_instr, opts.instrW2VDim, padding_idx=0) # not sure about the padding idx 
        #self.embs.weight.data.copy_(vec)

    def forward(self, x, sq_lengths):
        
        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x) 
        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)

        index_sorted_idx = sorted_idx\
                .view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the rnn
        out, hidden = self.lstm(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM
        # bi-directional
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
        output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
        output = output.view(output.size(0),output.size(1)*output.size(2))

        return output



class ingRNN(nn.Module):
    def __init__(self):
        super(ingRNN, self).__init__()
        self.irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)
        # _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        num_ing = 2343
        self.embs = nn.Embedding(num_ing, opts.ingrW2VDim, padding_idx=0) # not sure about the padding idx 
        #self.embs.weight.data.copy_(vec)

    def forward(self, x, sq_lengths):
        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x) 

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx\
                .view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the rnn
        out, hidden = self.irnn(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM
        # bi-directional
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
        output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
        output = output.view(output.size(0),output.size(1)*output.size(2))

        return output


class textEncoder(nn.Module):
    def __init__(self):
        super(textEncoder, self).__init__()

        self.stRNN_ = instrRNN()
        self.ingRNN_ = ingRNN()

        self.table = TableModule()
        self.recipe_embedding = nn.Sequential(
                nn.Linear(opts.irnnDim*2 + opts.srnnDim*2, opts.embDim, opts.embDim),
                nn.Tanh(),
            )

    def forward(self, y1, y2, z1, z2):
        recipe_emb = self.table([self.stRNN_(y1,y2),self.ingRNN_(z1,z2) ],1) # joining on the last dim 
        recipe_emb = self.recipe_embedding(recipe_emb)
        output = norm(recipe_emb)

        return output