import torch.nn as nn
from layers import HyperbolicGraphConvolution, FullyHyperbolicGraphConvolution
from manifolds.lorentz import Lorentz


class HGCN(nn.Module):
    def __init__(self, manifold: Lorentz, args):
        super(HGCN, self).__init__()
        self.manifold = manifold
        assert args.num_layers > 1
        hgc_layers = [HyperbolicGraphConvolution(self.manifold, args.network, args.num_layers)]
        self.layers = nn.Sequential(*hgc_layers)

    def encode(self, x, adj):
        inputs = (x, adj)
        output, _ = self.layers.forward(inputs)
        return output


class FHGCN(nn.Module):
    def __init__(self, manifold: Lorentz, args):
        super(FHGCN, self).__init__()
        self.manifold = manifold
        assert args.num_layers > 1
        hgc_layers = [FullyHyperbolicGraphConvolution(self.manifold, args.network, args.num_layers)]
        self.layers = nn.Sequential(*hgc_layers)

    def encode(self, x, adj):
        inputs = (x, adj)
        output, _ = self.layers.forward(inputs)
        return output
