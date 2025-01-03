import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from manifolds.lorentz import Lorentz


class FullyHyperbolicGraphConvolution(nn.Module):
    """
    Fully Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold: Lorentz, network, num_layers):
        super(FullyHyperbolicGraphConvolution, self).__init__()
        self.agg = FullyHypAgg(manifold, network, num_layers)

    def forward(self, inputs):
        x, adj = inputs
        h = self.agg.forward(x, adj)
        output = h, adj
        return output


class FullyStackGCNs(Module):
    def __init__(self, num_layers, manifold):
        super(FullyStackGCNs, self).__init__()
        self.n_layers = num_layers - 1
        self.manifold = manifold

    def fully_Agg(self, support):
        denom = (-self.manifold.inner(support, support, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        return support / denom

    def plainGCN(self, inputs):
        x, adj = inputs
        output = [x]
        for i in range(self.n_layers):
            support = torch.spmm(adj, output[i])
            output.append(self.fully_Agg(support))
        return output[-1]

    def resSumGCN(self, inputs):
        x, adj = inputs
        output = [x]
        for i in range(self.n_layers):
            support = torch.spmm(adj, output[i])
            output.append(self.fully_Agg(support))
        return self.fully_Agg(sum(output[1:]))

    def resAddGCN(self, inputs):
        x, adj = inputs
        output = [x]
        if self.n_layers == 1:
            support = torch.spmm(adj, x)
            return self.fully_Agg(support)
        for i in range(self.n_layers):
            if i == 0:
                support = torch.spmm(adj, output[i])
            else:
                support = output[i] + torch.spmm(adj, output[i])
            output.append(self.fully_Agg(support))
        return output[-1]

    def denseGCN(self, inputs):
        x, adj = inputs
        output = [x]
        for i in range(self.n_layers):
            if i > 0:
                support = sum(output[1:i + 1]) + torch.spmm(adj, output[i])
            else:
                support = torch.spmm(adj, output[i])
            output.append(self.fully_Agg(support))
        return output[-1]

    def lightGCN(self, inputs):
        x, adj = inputs
        output = [x]
        for i in range(self.n_layers):
            support = torch.spmm(adj, output[i])
            output.append(self.fully_Agg(support))
        return self.fully_Agg(torch.stack(output, dim=1).mean(dim=1))


class FullyHypAgg(Module):
    """
    Fully Hyperbolic aggregation layer.
    """

    def __init__(self, manifold: Lorentz, network, num_layers):
        super(FullyHypAgg, self).__init__()
        self.manifold = manifold
        self.fullyStackGCNs = getattr(FullyStackGCNs(num_layers, self.manifold), network)

    def forward(self, x, adj):
        output = self.fullyStackGCNs((x, adj))
        return output

    def extra_repr(self):
        return 'c={}'.format(self.manifold.k)
