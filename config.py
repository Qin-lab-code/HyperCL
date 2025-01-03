import argparse
from utils.util import add_flags_from_config

config_args = {
    'training_config': {
        'log': ('cd', 'None for no logging'),
        'lr': (0.002, 'learning rate'),
        'weight-decay': (0.005, 'l2 regularization strength'),
        'momentum': (0.95, 'momentum in optimizer'),
        'epochs': (500, 'maximum number of epochs to train for'),
        'batch-size': (10000, 'batch size'),
        'seed': (1234, 'seed for data split and training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'eval_batch_num': (0, 'val compute batch num'),
        'device': ('cuda:0', 'device')
    },
    'model_config': {
        'embedding_dim': (50, 'user item embedding dimension'),
        'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
        'scale': (0.1, 'scale'),
        'max_norm': (1.5, 'max norm'),
        'num-layers': (4,  'number of hidden layers in encoder'),
        'margin': (0.1, 'margin value in the metric learning loss'),
        'ssl_ratio': (0.1, ''),
        'ssl_temp': (0.05, 'ssl temp'),
        'ssl_reg': (0.005, 'ssl reg')
    },
    'data_config': {
        'dataset': ('Amazon-CD', 'which dataset to use'),
        'num_neg': (1, 'number of negative samples'),
        'test_ratio': (0.2, 'proportion of test edges for link prediction'),
        'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
