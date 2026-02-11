import math
import torch
import torch.nn.functional as F
from utils import rotations


class PoseNet(torch.nn.Module):
    def __init__(self, embed_size, layers_feat=(256, 256, 256, 256, 256, 256, 256, 256), skip=(4,), activ="relu"):
        super().__init__()

        layers_feat = [None, *layers_feat]

        # init network
        self.transNet = BaseNet(layers_feat, skip, activ, 'tanh', embed_size, 3,False)
        self.rotsNet = BaseNet(layers_feat, skip, activ, 'custom_tanh', embed_size, 4, True)

    def get_param_groups(self):
        return {
            'rotation': self.rotsNet.parameters(),
            'translation': self.transNet.parameters(),
            'rest': []
        }

    def forward(self, time, time_embed):
        trans_est = self.transNet.forward(time_embed)
        rots_feat_est = self.rotsNet.forward(time_embed)
        rotmat_est = rotations.quaternion_to_matrix(rots_feat_est)
        # make c2w
        c2w_est = torch.cat([rotmat_est, trans_est.unsqueeze(-1)], dim=-1)

        return c2w_est


class BaseNet(torch.nn.Module):
    def __init__(self, layers_feat, skip, activ, hidden_activ, in_channels, out_channels, normalize=False):
        super().__init__()

        self.layers_feat = layers_feat
        self.skip = skip
        self.normalize = normalize

        hidden_activ_dict = {
            'tanh': torch.tanh,
            'custom_tanh': self.custom_tanh,
        }
        self.hidden_activ = hidden_activ_dict[hidden_activ]
        self.activ = getattr(F, activ)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_network(out_channels)

    @staticmethod
    def custom_tanh(x):
        # See opensource PoseNet implementation
        x[..., 1:] = torch.tanh(x[..., 1:])
        x[..., 0] = 1 * (1 - torch.tanh(x[..., 0]))
        return x

    def define_network(self, out_channels):
        self.mlp = torch.nn.ModuleList()
        layers_list = self.layers_feat
        L = list(zip(layers_list[:-1], layers_list[1:]))

        for li, (k_in, k_out) in enumerate(L):
            if li == 0: k_in = self.in_channels
            if li in self.skip: k_in += self.in_channels
            if li == len(L) - 1: k_out = out_channels
            linear = torch.nn.Linear(k_in, k_out)
            self.initialize_weights(linear, out="small" if li == len(L) - 1 else "all")
            self.mlp.append(linear)

    def initialize_weights(self, linear, out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu")  # sqrt(2)
        if out == "all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out == "small":
            torch.nn.init.uniform_(linear.weight, b=1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight, gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, time_enc):
        x = time_enc
        for li, layer in enumerate(self.mlp):
            if li in self.skip: x = torch.cat([x, time_enc], dim=-1)
            x = layer(x)
            if li == len(self.mlp) - 1:
                x = self.hidden_activ(x)  # note we assume bounds is [-1,1]
            else:
                x = self.activ(x)

        if self.normalize:
            norm_x = torch.norm(x, dim=-1)
            x = x / (norm_x[..., None] + 1e-18)

        return x
