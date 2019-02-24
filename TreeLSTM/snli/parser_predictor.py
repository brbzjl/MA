import torch
import torch.nn as nn
from spinn_trans_pre import SPINN1

class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(*size, -1)


class Linear(Bottle, nn.Linear):
    pass


class BatchNorm(Bottle, nn.BatchNorm1d):
    pass


class Feature(nn.Module):

    def __init__(self, size, dropout):
        super(Feature, self).__init__()
        self.bn = nn.BatchNorm1d(size * 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, prem, hypo):
        return self.dropout(self.bn(torch.cat(
            [prem, hypo, prem - hypo, prem * hypo], 1)))


class Parser(nn.Module):

    def __init__(self, config):
        super(Parser, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)#word length x 300
        self.projection = Linear(config.d_embed, config.d_proj)#300 is projected to 600
        self.embed_bn = BatchNorm(config.d_proj)
        self.embed_dropout = nn.Dropout(p=config.embed_dropout)#0.08
        self.encoder = SPINN1(config)
        # feat_in_size = config.d_hidden * (
        #     2 if self.config.birnn and not self.config.spinn else 1)
        # self.feature = Feature(feat_in_size, config.mlp_dropout)
        # self.mlp_dropout = nn.Dropout(p=config.mlp_dropout)#0.07
        # self.relu = nn.ReLU()
        # mlp_in_size = 4 * feat_in_size
        # mlp = [nn.Linear(mlp_in_size, config.d_mlp), self.relu,
        #        nn.BatchNorm1d(config.d_mlp), self.mlp_dropout]
        # for i in range(config.n_mlp_layers - 1):
        #     mlp.extend([nn.Linear(config.d_mlp, config.d_mlp), self.relu,
        #                 nn.BatchNorm1d(config.d_mlp), self.mlp_dropout])
        # mlp.append(nn.Linear(config.d_mlp, config.d_out))
        # #self.out = nn.ModuleList(mlp)
        # self.out = nn.Sequential(*mlp)
    def forward(self, batch):
        # import pdb
        # pdb.set_trace()
        #print(batch.premise)
        # the size of the inputs
        # the first dim is 51 the sequence itself,  the length of sequence(words)
        # the second dim is the batch size 128
        # the third is indexes elements of the input 300 the embedding vectors length of each words

        prem_embed = self.embed(batch.premise[0])
        prem_embed = self.projection(prem_embed)  # no relu
        prem_embed = self.embed_dropout(self.embed_bn(prem_embed))

        if hasattr(batch, 'premise_transitions'):
            prem_trans = batch.premise_transitions
        else:
            prem_trans = None
        trans_pred = self.encoder(prem_embed, prem_trans)#128x300,

        #scores = self.out(self.feature(premise, hypothesis))
        #print(premise[0][:5], hypothesis[0][:5])
        return trans_pred
