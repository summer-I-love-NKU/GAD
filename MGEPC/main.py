import os
import random
from GetData import *
from Train import *
from models import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp')
parser.add_argument('--model', type=str, default='MGEPC')
parser.add_argument('--topk', type=int, default=4)
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--drop_ratio', type=float, default=0.1)
parser.add_argument('--train_ratio', type=float, default=0.4)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--order', type=int, default=2)
parser.add_argument('--homo', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.5)
args = parser.parse_args()


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(seed=args.seed)

def main(args):
    print("START!!!!!!!!!")
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    graph = Dataset(dataset_name, homo).graph

    from dgl.nn.pytorch.factory import KNNGraph

    in_feats = graph.ndata['feature'].shape[1]
    print(f'in_feats dim={in_feats}')

    num_classes = 2

    print(args.dataset)
    print(graph)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    knn_type='naive_knn'
    if knn_type=='naive_knn':
        kg = KNNGraph(args.topk)
        g = kg(graph.ndata['feature'].cpu())
        g = dgl.to_bidirected(g)
        g.ndata['feature'] = graph.ndata['feature'].to('cpu')
        g = g.to(device)
    elif knn_type=='chunk_knn':
        kg = KNNGraph(args.topk)
        features = graph.ndata['feature']
        num_nodes = features.shape[0]
        chunk_size = args.chunk_size
        print('chunk_size = ',chunk_size)
        edges = []
        for i in range(0, num_nodes, chunk_size):
            end = min(i + chunk_size, num_nodes)
            sub_features = features[i:end]
            sub_g = kg(sub_features)
            sub_g = sub_g.cpu()
            sub_g = dgl.to_bidirected(sub_g)
            sub_g = sub_g.to(device)
            sub_edges = sub_g.edges()
            sub_edges = (sub_edges[0] + i, sub_edges[1] + i)
            edges.append(sub_edges)

        src_edges = torch.cat([edge[0] for edge in edges])
        dst_edges = torch.cat([edge[1] for edge in edges])
        g = dgl.graph((src_edges, dst_edges), num_nodes=num_nodes)
        g.ndata['feature'] = graph.ndata['feature']
        g = g.to(device)
    else:
        print('error')
        exit(0)

    model = MGEPC(in_feats, h_feats, num_classes, graph, g, args, d=order)
    model = model.to(device)

    mf1, auc, auc_pr = train(model, graph, args)

    print(f"Final result: MacroF1 {mf1:.2f} AUC_ROC {auc:.2f}  AUC_PR {auc_pr:.2f} ")


main(args)