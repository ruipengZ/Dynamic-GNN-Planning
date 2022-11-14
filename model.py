import numpy as np
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn.conv import MessagePassing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(MPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(Lin(embed_size * 4, embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x, edge_index, edge_attr):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        return torch.max(x, out)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_j - x_i, x_j, x_i, edge_attr], dim=-1)
        values = self.fx(z)
        return values

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)


class GNNet(torch.nn.Module):
    def __init__(self, config_size, embed_size, obs_size, use_obstacles=True):
        super(GNNet, self).__init__()

        self.config_size = config_size
        self.embed_size = embed_size
        self.use_obstacles = use_obstacles
        self.obs_size = obs_size

        # label:1 goal/source
        self.hx = Seq(Lin((config_size + 1) * 4, embed_size),
                      ReLU(),
                      Lin(embed_size, embed_size))
        self.hy = Seq(Lin((config_size + 1) * 3, embed_size),
                      ReLU(),
                      Lin(embed_size, embed_size))
        self.mpnn = MPNN(embed_size)

        self.obs_node_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.obs_edge_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.node_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        self.edge_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])

        self.fy = Seq(Lin(embed_size * 3, embed_size), ReLU(),
                      Lin(embed_size, embed_size))


    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        for op in self.ops:
            op.reset_parameters()
        self.node_feature.reset_parameters()
        self.edge_feature.reset_parameters()


    def forward(self, v, labels, obstacles, pos_enc, edge_index, loop, **kwargs):

        self.labels = labels
        # labels: ?goal, one-hot
        v = torch.cat((v, labels), dim=-1)
        goal = v[labels[:, 0] == 1].view(1, -1)
        x = self.hx(torch.cat((v, goal.repeat(len(v), 1), v - goal, (v - goal) ** 2), dim=-1))  # node

        vi, vj = v[edge_index[0, :]], v[edge_index[1, :]]
        y = self.hy(torch.cat((vj - vi, vj, vi), dim=-1))  # edge

        if self.use_obstacles:

            obs_node_code = self.obs_node_code(obstacles.view(-1, self.obs_size))
            obs_node_code = obs_node_code + pos_enc

            obs_edge_code = self.obs_edge_code(obstacles.view(-1, self.obs_size))
            obs_edge_code = obs_edge_code + pos_enc

            for na, ea in zip(self.node_attentions, self.edge_attentions):
                x = na(x, obs_node_code)
                y = ea(y, obs_edge_code)

        # message passing loops
        for _ in range(loop):
            x = self.mpnn(x, edge_index, y)
            xi, xj = x[edge_index[0, :]], x[edge_index[1, :]]
            y = torch.max(y, self.fy(torch.cat((xj - xi, xj, xi), dim=-1)))

        edge_feat = y.new_zeros(len(v), len(v), self.embed_size)
        edge_feat[edge_index[0, :], edge_index[1, :]] = y

        return edge_feat, x


class PolicyHead(torch.nn.Module):
    def __init__(self, embed_size, obs_size=9):
        super(PolicyHead, self).__init__()
        self.obs_size = obs_size

        self.layer1 = Seq(Lin(embed_size + obs_size * 5, embed_size*2), ReLU(),
                          Lin(embed_size*2, embed_size))
        self.layer2 = Seq(Lin(embed_size, embed_size), ReLU(),
                          Lin(embed_size, 1, bias=False))

    def forward(self, edge_feat, obs, pe):
        obs_code = obs + pe
        policy = self.layer1(torch.concat([edge_feat, obs_code], dim=-1))
        policy = self.layer2(policy)
        return policy



class Attention(torch.nn.Module):

    def __init__(self, embed_size, temperature):
        super(Attention, self).__init__()
        self.temperature = temperature
        self.embed_size = embed_size
        self.key = Lin(embed_size, embed_size, bias=False)
        self.query = Lin(embed_size, embed_size, bias=False)
        self.value = Lin(embed_size, embed_size, bias=False)
        self.layer_norm = torch.nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, v_code, obs_code):
        v_value = self.value(v_code)
        obs_value = self.value(obs_code)

        v_query = self.query(v_code)

        v_key = self.key(v_code)
        obs_key = self.key(obs_code)

        obs_attention = (v_query @ obs_key.T)
        self_attention = (v_query.reshape(-1) * v_key.reshape(-1)).reshape(-1, self.embed_size).sum(dim=-1)
        whole_attention = torch.cat((self_attention.unsqueeze(-1), obs_attention), dim=-1)
        whole_attention = (whole_attention / self.temperature).softmax(dim=-1)

        v_code_new = (whole_attention.unsqueeze(-1) *
                        torch.cat((v_value.unsqueeze(1), obs_value.unsqueeze(0).repeat(len(v_code), 1, 1)),
                                  dim=1)).sum(dim=1)

        return self.layer_norm(v_code_new + v_code)


class FeedForward(torch.nn.Module):
    def __init__(self, d_in, d_hid):
        super(FeedForward, self).__init__()
        self.w_1 = Lin(d_in, d_hid)  # position-wise
        self.w_2 = Lin(d_hid, d_in)  # position-wise
        self.layer_norm = torch.nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.w_2((self.w_1(x)).relu())
        x += residual
        x = self.layer_norm(x)

        return x


class Block(torch.nn.Module):
    def __init__(self, embed_size):
        super(Block, self).__init__()
        self.attention = Attention(embed_size, embed_size ** 0.5)
        self.v_feed = FeedForward(embed_size, embed_size)

    def forward(self, v_code, obs_code):
        v_code = self.attention(v_code, obs_code)
        v_code = self.v_feed(v_code)

        return v_code



class TemporalEncoder(nn.Module):
    def __init__(self, embed_size, min_freq=1e-4, max_seq_len=1024):
        super().__init__()
        self.embed_size = embed_size

        position = torch.arange(max_seq_len)
        freqs = min_freq ** (2 * (torch.arange(self.embed_size) // 2) / self.embed_size)
        tmp_enc = position.reshape(-1, 1) * freqs.reshape(1, -1)
        tmp_enc[:, ::2] = np.cos(tmp_enc[:, ::2])
        tmp_enc[:, 1::2] = np.sin(tmp_enc[:, 1::2])

        self.register_buffer('tmp_enc', tmp_enc)

    def forward(self, time):
        x = self.tmp_enc[time, :]
        return x