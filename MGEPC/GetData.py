import dgl
from dgl.data import FraudYelpDataset, FraudAmazonDataset,CoraGraphDataset,PubmedGraphDataset
from dgl.data.utils import load_graphs
import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.unicode_minus']=False
from dgl.nn.pytorch.conv import EdgeWeightNorm
import dgl.function as fn
warnings.filterwarnings("ignore")


def random_walk_update(graph, delete_ratio, adj_type):
    edge_weight = torch.ones(graph.num_edges())
    if adj_type == 'sym':
        norm = EdgeWeightNorm(norm='both')
    else:
        norm = EdgeWeightNorm(norm='left')
    graph.edata['w'] = norm(graph, edge_weight)
    # functions
    aggregate_fn = fn.u_mul_e('h', 'w', 'm')
    reduce_fn = fn.sum(msg='m', out='ay')

    graph.ndata['h'] = graph.ndata['pred_y']
    graph.update_all(aggregate_fn, reduce_fn)
    graph.ndata['ly'] = graph.ndata['pred_y'] - graph.ndata['ay']
    # graph.ndata['lyyl'] = torch.matmul(graph.ndata['ly'], graph.ndata['ly'].T)
    graph.apply_edges(inner_product_black)
    # graph.apply_edges(inner_product_white)
    black = graph.edata['inner_black']
    # white = graph.edata['inner_white']
    # delete
    threshold = int(delete_ratio * graph.num_edges())
    edge_to_move = set(black.sort()[1][:threshold].tolist())
    # edge_to_protect = set(white.sort()[1][-threshold:].tolist())
    edge_to_protect = set()
    graph_new = dgl.remove_edges(graph, list(edge_to_move.difference(edge_to_protect)))
    return graph_new

def inner_product_black(edges):
    return {'inner_black': (edges.src['ly'] * edges.dst['ly']).sum(axis=1)}

def inner_product_white(edges):
    return {'inner_white': (edges.src['ay'] * edges.dst['ay']).sum(axis=1)}

def find_inter(edges):
    return edges.src['label'] != edges.dst['label'] 

def cal_hetero(edges):
    return {'same': edges.src['label'] != edges.dst['label']}

def cal_hetero_normal(edges):
    return {'same_normal': (edges.src['label'] != edges.dst['label']) & (edges.src['label'] == 0)}

def cal_normal(edges):
    return {'normal': edges.src['label'] == 0}

def cal_hetero_anomal(edges):
    return {'same_anomal': (edges.src['label'] != edges.dst['label']) & (edges.src['label'] == 1)}

def cal_anomal(edges):
    return {'anomal': edges.src['label'] == 1}




raw_dir='./dataset/'
class Dataset:
    def __init__(self, name='yelp', homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None
        if name == 'yelp':
            dataset = FraudYelpDataset(raw_dir=raw_dir,force_reload=False)
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset(raw_dir=raw_dir)
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'pubmed':
            dataset = PubmedGraphDataset(raw_dir=raw_dir)
            graph = dgl.to_homogeneous(dataset[0], ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
            labels = graph.ndata['label']
            normal_idx = np.where(labels != 0)[0]
            abnormal_idx = np.where(labels == 0)[0]
            labels[normal_idx] = 0
            labels[abnormal_idx] = 1
            graph.ndata['label'] = labels
            graph.ndata['feature'] = graph.ndata['feat']


        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()

        self.graph = graph