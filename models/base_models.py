import numpy as np
import torch
import torch.nn as nn
from manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter
from models.encoders import HGCN, FHGCN
from utils.util import ndcg_func, recall_func, recall_single_func, ndcg_single_func


class HyperCL(nn.Module):
    def __init__(self, users_items, args):
        super(HyperCL, self).__init__()
        self.device = args.device
        self.manifold = Lorentz(max_norm=args.max_norm)
        self.nnodes = args.n_nodes
        self.encoder1 = HGCN(self.manifold, args)
        self.encoder2 = FHGCN(self.manifold, args)

        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.num_layers = args.num_layers
        self.latent_dim = args.embedding_dim
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.args = args

        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items,
                                      embedding_dim=args.embedding_dim).to(self.device)
        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], project=True))
        self.embedding.weight = ManifoldParameter(self.embedding.weight, self.manifold, True)

    def encode(self, adj, adj_1, adj_2):
        x = self.embedding.weight
        if torch.cuda.is_available():
            adj = adj.to(self.device)
            adj_1 = adj_1.to(self.device)
            adj_2 = adj_2.to(self.device)
            x = x.to(self.device)
        h1 = self.encoder1.encode(x, adj)
        h2 = self.encoder1.encode(x, adj_1)
        h3 = self.encoder2.encode(x, adj_2)
        return h1, h2, h3

    def encode4eval(self, adj):
        x = self.embedding.weight
        if torch.cuda.is_available():
            adj = adj.to(self.device)
            x = x.to(self.device)
        h = self.encoder1.encode(x, adj)
        return h

    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out)
        return sqdist

    def Contra_loss(self, u_h, i_h, u_f, i_f):
        pos_score_user = -1 * self.manifold.sqdist(u_h, u_f)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        tot_score_user = -1 * self.manifold.sqdist_multi(u_h, u_f)
        tot_score_user = torch.exp(tot_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / tot_score_user).sum()

        pos_score_item = -1 * self.manifold.sqdist(i_h, i_f)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        tot_score_item = -1 * self.manifold.sqdist_multi(i_h, i_f)
        tot_score_item = torch.exp(tot_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / tot_score_item).sum()
        return ssl_loss_user + ssl_loss_item

    def compute_loss(self, embeddings1, embeddings2, embeddings3, triples):
        train_edges = triples[:, [0, 1]]
        sampled_false_edges = triples[:, [0, 2]]
        pos_scores = self.decode(embeddings1, train_edges)
        neg_scores = self.decode(embeddings1, sampled_false_edges)
        loss = pos_scores - neg_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)

        u_embeddings_h = embeddings2[train_edges[:, 0], :]
        pos_embeddings_h = embeddings2[train_edges[:, 1], :]
        u_embeddings_f = embeddings3[train_edges[:, 0], :]
        pos_embeddings_f = embeddings3[train_edges[:, 1], :]
        ssl_loss = self.Contra_loss(u_embeddings_h, pos_embeddings_h, u_embeddings_f, pos_embeddings_f)
        return loss + self.ssl_reg * ssl_loss

    def predict(self, h, train_csr, test_csr, test_dict, eval_batch_num=0):
        if eval_batch_num == 0:
            Recall, Ndcg = {k: [] for k in [10, 20]}, {k: [] for k in [10, 20]}
            for i in range(self.num_users):
                embu = h[i, :].repeat(self.num_items).view(self.num_items, -1)
                embi = h[np.arange(self.num_users, self.num_users + self.num_items), :]
                probs = (-1 * self.manifold.sqdist(embu, embi)).detach().cpu().numpy()
                probs[train_csr.getrow(i).nonzero()[1]] = np.NINF
                ind = np.argpartition(probs, -20)[-20:]
                arr_ind = probs[ind]
                arr_ind_argsort = np.argsort(arr_ind)[::-1]
                pred_list = ind[arr_ind_argsort]
                user_actual = test_csr.getrow(i).nonzero()[1].tolist()
                all_ndcg = ndcg_single_func(user_actual, pred_list)
                for k in [10, 20]:
                    Recall[k].append(recall_single_func(user_actual, pred_list, k))
                    Ndcg[k].append(all_ndcg[k - 1])
            avg_recall = {k: np.mean(v) for k, v in Recall.items()}
            avg_ndcg = {k: np.mean(v) for k, v in Ndcg.items()}
            return avg_recall, avg_ndcg
        recall, ndcg = {}, {}
        item = h[np.arange(self.num_users, self.num_users + self.num_items), :]
        if eval_batch_num == 1:
            user = h[np.arange(self.num_users), :]
            probs = (-1 * self.manifold.sqdist_multi(user, item)).detach().cpu().numpy()
        else:
            batch_size = (self.num_users // eval_batch_num) + 1
            all_probs = []
            for start in range(0, self.num_users, batch_size):
                end = min(start + batch_size, self.num_users)
                user_batch = h[np.arange(start, end), :]
                probs_batch = (-1 * self.manifold.sqdist_multi(user_batch, item)).detach().cpu().numpy()
                all_probs.append(probs_batch)
            probs = np.concatenate(all_probs, axis=0)
        probs[train_csr.nonzero()] = np.NINF
        ind = np.argpartition(probs, -20)[:, -20:]
        arr_ind = probs[np.arange(len(probs))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(probs)), ::-1]
        pred_list = ind[np.arange(len(probs))[:, None], arr_ind_argsort]
        all_ndcg = ndcg_func([*test_dict.values()], pred_list)
        for k in [10, 20]:
            recall[k] = recall_func(test_dict, pred_list, k)
            ndcg[k] = all_ndcg[k - 1]
        return recall, ndcg
