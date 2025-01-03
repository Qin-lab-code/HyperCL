import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from manifolds.lorentz import Lorentz


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold: Lorentz, network, num_layers):
        super(HyperbolicGraphConvolution, self).__init__()
        self.agg = HypAgg(manifold, network, num_layers)

    def forward(self, inputs):
        x, adj = inputs
        h = self.agg.forward(x, adj)
        output = h, adj
        return output


class StackGCNs(Module):

    def __init__(self, num_layers):
        super(StackGCNs, self).__init__()
        self.num_gcn_layers = num_layers - 1

    def plainGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def resSumGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return sum(output[1:])

    def resAddGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        if self.num_gcn_layers == 1:
            return torch.spmm(adj, x_tangent)
        for i in range(self.num_gcn_layers):
            if i == 0:
                output.append(torch.spmm(adj, output[i]))
            else:
                output.append(output[i] + torch.spmm(adj, output[i]))
        return output[-1]

    def denseGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            if i > 0:
                output.append(sum(output[1:i + 1]) + torch.spmm(adj, output[i]))
            else:
                output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def lightGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return torch.stack(output, dim=1).mean(dim=1)


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold: Lorentz, network, num_layers):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.stackGCNs = getattr(StackGCNs(num_layers), network)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x)
        output = self.stackGCNs((x_tangent, adj))
        output = self.manifold.expmap0(output, project=True)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.manifold.c)
