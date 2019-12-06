import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn import functional as F
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

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.u = torch.nn.Parameter(torch.randn(input_dim)) # u = [2*hid_dim]
        self.u.requires_grad = True
        self.fc = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        # x = [BS, num_vec, 2*hid_dim]
        mask = (x!=0)
        # a trick used to find the mask for the softmax
        mask = mask[:,:,0]
        h = torch.tanh(self.fc(x)) # h = [BS, num_vec, 2*hid_dim]
        tmp = h.matmul(self.u) # tmp = [BS, num_vec], unnormalized importance
        masked_tmp = tmp.masked_fill((~ mask).byte(), -1e32)
        alpha = F.softmax(masked_tmp, dim=1) # alpha = [BS, num_vec], normalized importance
        alpha = alpha.unsqueeze(-1) # alpha = [BS, num_vec, 1]
        out = x * alpha # out = [BS, num_vec, 2*hid_dim]
        out = out.sum(dim=1) # out = [BS, 2*hid_dim]
        return out

class instrRNN(nn.Module):
    def __init__(self, using_att=False):
        super(instrRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=opts.instrW2VDim, hidden_size=opts.srnnDim, bidirectional=True, batch_first=True)
        #_, vec = torchwordemb.load_word2vec_bin(opts.instrW2V)
        num_instr = 13
        self.embs = nn.Embedding(num_instr, opts.instrW2VDim, padding_idx=0) # not sure about the padding idx 
        #self.embs.weight.data.copy_(vec)
        self.using_att = using_att
        if self.using_att:
            self.attention_layer = AttentionLayer(2*opts.srnnDim)

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

        if self.using_att:
            hidden = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

            # unsort the output
            _, original_idx = sorted_idx.sort(0, descending=False)

            # LSTM
            # bi-directional
            unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
            # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
            output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
            output = self.attention_layer(output)

        else:
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
    def __init__(self, using_att=False):
        super(ingRNN, self).__init__()
        self.irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)
        # _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        num_ing = 2343
        self.embs = nn.Embedding(num_ing, opts.ingrW2VDim, padding_idx=0) # not sure about the padding idx 
        #self.embs.weight.data.copy_(vec)
        self.using_att = using_att
        if self.using_att:
            self.attention_layer = AttentionLayer(2*opts.irnnDim)

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

        if self.using_att:
            hidden = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # unsort the output
            _, original_idx = sorted_idx.sort(0, descending=False)
            # LSTM
            # bi-directional
            unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
            # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
            output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
            output = self.attention_layer(output)

        else:
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
    def __init__(self, using_instrs=True, using_att=False):
        super(textEncoder, self).__init__()

        self.using_instrs = using_instrs
        self.using_att = using_att
        self.ingRNN_ = ingRNN(using_att=self.using_att)

        self.recipe_embedding = nn.Sequential(
                nn.Linear(opts.irnnDim*2, opts.embDim, opts.embDim),
                nn.Tanh(),
            )

        if using_instrs:
            self.stRNN_ = instrRNN(using_att=self.using_att)
            self.recipe_embedding = nn.Sequential(
                nn.Linear(opts.irnnDim*2 + opts.srnnDim*2, opts.embDim, opts.embDim),
                nn.Tanh(),
            )
            self.table = TableModule()

    def forward(self, y1, y2, z1, z2):
        if self.using_instrs:
            recipe_emb = self.table([self.stRNN_(y1,y2),self.ingRNN_(z1,z2) ],1) # joining on the last dim 
            recipe_emb = self.recipe_embedding(recipe_emb)
            output = norm(recipe_emb)
        else:
            recipe_emb = self.recipe_embedding(self.ingRNN_(z1,z2))
            output = norm(recipe_emb)
        return output