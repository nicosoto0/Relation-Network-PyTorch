import torch
import torch.nn as nn
from src.models.MLP import MLP

class RelationNetwork(nn.Module):

    def __init__(self, query_dim, hidden_dims_g, output_dim_g, drops_g, drop_prob_g, hidden_dims_f, output_dim_f, drops_f, drop_prob_f, batch_size, device):
        '''
        :param object_dim: Equal to LSTM hidden dim. Dimension of the single object to be taken into consideration from g.
        '''

        super(RelationNetwork, self).__init__()

        self.query_dim = query_dim
        self.input_dim_g = self.query_dim # this changed for LSTM

        self.hidden_dims_g = hidden_dims_g
        self.output_dim_g = output_dim_g
        self.drops_g = drops_g
        self.drop_prob_g = drop_prob_g
        
        self.input_dim_f = self.output_dim_g
        self.hidden_dims_f = hidden_dims_f
        self.output_dim_f = output_dim_f
        self.drops_f = drops_f
        self.drop_prob_f = drop_prob_f

        self.batch_size = batch_size
        self.device = device

        self.g = MLP(self.input_dim_g, self.hidden_dims_g, self.output_dim_g, self.drops_g, nonlinear=True, drop_prob=self.drop_prob_g)
        self.f = MLP(self.input_dim_f, self.hidden_dims_f, self.output_dim_f, self.drops_f, nonlinear=True, drop_prob=self.drop_prob_f)

    def forward(self, q=None):#, objectNums=0):
        '''
        :param q: (batch, length_q) query, optional.
        '''
        g   = self.g(q)
        out = self.f(g) # (output_dim_f)

        return out
