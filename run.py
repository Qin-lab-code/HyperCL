import time
import traceback
from datetime import datetime
import numpy as np
import torch

from config import parser
from models.base_models import HyperCL
from optim import RiemannianSGD
from utils.data import Data
from utils.util import set_seed, sp_mat_to_sp_tensor
from utils.log import Logger
from utils.sampler import WarpSampler


def train(model):
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum)

    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Total number of parameters: {tot_params}")

    num_pairs = data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    print(num_batches)

    # === Train model
    for epoch in range(1, args.epochs + 1):
        avg_loss = 0.
        # === batch training
        t_start = time.time()
        sub_graph1 = sp_mat_to_sp_tensor(data.create_adj_mat(True, 'ed')).to(args.device)
        sub_graph2 = sp_mat_to_sp_tensor(data.create_adj_mat(True, 'ed')).to(args.device)
        t_graph = time.time() - t_start

        for batch in range(num_batches):
            triples = sampler.next_batch()
            model.train()
            optimizer.zero_grad()
            embeddings1, embeddings2, embeddings3 = model.encode(data.adj_train_norm, sub_graph1, sub_graph2)
            train_loss = model.compute_loss(embeddings1, embeddings2, embeddings3, triples)
            train_loss.backward()
            optimizer.step()
            avg_loss += train_loss / num_batches
        t_train = time.time() - t_start

        # === evaluate at the end of each batch
        avg_loss = avg_loss.detach().cpu().numpy()
        log.write('Train:{:3d} {:.5f} {:f} {:f}\n'.format(epoch, avg_loss, t_graph, t_train))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                t_eval_start = time.time()
                embeddings = model.encode4eval(data.adj_train_norm)
                t_eval_encode = time.time() - t_eval_start
                recall, ndcg = model.predict(embeddings, data.train_csr, data.test_csr, data.test_dict, args.eval_batch_num)
                t_eval = time.time() - t_eval_start
                log.write('Test:{:3d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:f}\t{:f}\n'.
                          format(epoch + 1, recall[10], recall[20], ndcg[10], ndcg[20], t_eval_encode, t_eval))
    sampler.close()


if __name__ == '__main__':
    args = parser.parse_args()

    if args.log:
        now = datetime.now()
        now = now.strftime('%m-%d_%H-%M-%S')
        log = Logger(args.log, now)

        for arg in vars(args):
            log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    else:
        print(args.dim, args.lr, args.weight_decay, args.margin, args.batch_size)
        print(args.scale, args.num_layers, args.network)

    # === fix seed
    set_seed(args.seed)

    # === prepare data
    data = Data(args.dataset, args.norm_adj, args.seed, args.test_ratio, args.ssl_ratio)
    total_edges = data.adj_train.count_nonzero()
    args.n_nodes = data.num_users + data.num_items
    args.feat_dim = args.embedding_dim

    # === negative sampler (iterator)
    sampler = WarpSampler((data.num_users, data.num_items), data.adj_train, args.batch_size, args.num_neg)

    model = HyperCL((data.num_users, data.num_items), args)
    model = model.to(args.device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print('model is running on', next(model.parameters()).device)

    try:
        train(model)
    except Exception:
        sampler.close()
        traceback.print_exc()
