from model import Model,DGI
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='ANEMONE')
parser.add_argument('--expid', type=int)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio_patch', type=int, default=1)
parser.add_argument('--negsamp_ratio_context', type=int, default=1)
parser.add_argument('--alpha_content', type=float, default=1.0, help='how much context-level involves')
parser.add_argument('--alpha', type=float, default=0.5, help='how much loval-level involves for training')
parser.add_argument('--beta',  type=float, default=0.5, help='how much local-level involves for predicting')
args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = [i + 1 for i in range(args.runs)]

    batch_size = args.batch_size
    subgraph_size = args.subgraph_size

    adj, features, labels, idx_train, idx_val,\
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

    features, _ = preprocess_features(features)
    dgl_graph = adj_to_dgl_graph(adj)

    num_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    all_auc = []
    for run in range(args.runs):
        seed = seeds[run]
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        batch_num = num_nodes // batch_size + 1

        
        
        # Testing
        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context,
                      args.readout).to(device)
        dgi_model = DGI(ft_size, args.embedding_dim, 'prelu').to(device)
        model.load_state_dict(torch.load(f'best_cola_{args.dataset}.pkl'))
        dgi_model.load_state_dict(torch.load(f'best_dgi_{args.dataset}.pkl'))

        multi_round_ano_score = np.zeros((args.auc_test_rounds, num_nodes))
        print('Testing AUC!', flush=True)

        for round in range(args.auc_test_rounds):
            print(f'round={round+1}')
            all_idx = list(range(num_nodes))
            random.shuffle(all_idx)
            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
            for batch_idx in range(batch_num):
                
                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]
                cur_batch_size = len(idx)
                ba = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                with torch.no_grad():
                    test_logits_1, test_logits_2 = model(bf, ba)
                    test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                    test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))

                if args.alpha_content != 1.0 and args.alpha_content != 0.0:
                    if args.negsamp_ratio_context == 1 and args.negsamp_ratio_patch == 1:
                        ano_score_1 = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                        ano_score_2 = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score_1 = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                            cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()  # context
                        ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                            cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()  # patch
                    ano_score = args.alpha_content * ano_score_1 + (1 - args.alpha_content) * ano_score_2
                elif args.alpha_content == 1.0:
                    if args.negsamp_ratio_context == 1:
                        ano_score = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()  # context
                elif args.alpha_content == 0.0:
                    if args.negsamp_ratio_patch == 1:
                        ano_score = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()  # patch

                multi_round_ano_score[round, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)

        with torch.no_grad():
            idx = np.random.permutation(num_nodes)
            shuf_features = features[:, idx, :]

            if torch.cuda.is_available():
                shuf_features = shuf_features.cuda()

            logits = dgi_model(features, shuf_features, adj, False, None, None, None)
            logits = torch.sigmoid(torch.squeeze(logits))
            ano_score_dgi = - (logits[:num_nodes] - logits[num_nodes:]).cpu().numpy()

        ano_score_final = args.beta * ano_score_final + (1-args.beta) * ano_score_dgi

        auc = roc_auc_score(ano_label, ano_score_final)
        all_auc.append(auc)
        print('Testing AUC:{:.4f}'.format(auc), flush=True)

    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')