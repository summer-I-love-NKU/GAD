import torch
import torch.nn.functional as F
import time

from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score,precision_recall_curve,auc
from sklearn.model_selection import train_test_split
import numpy as np


def train(model, g, args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model=model.to(device)
    features = g.ndata['feature'].to(device)
    labels = g.ndata['label'].to('cpu').numpy()# train_test_split 必须在cpu

    index = list(range(len(labels)))
    dataset_name = args.dataset
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    


    idx_train = torch.tensor(idx_train, device=device)
    idx_valid = torch.tensor(idx_valid, device=device)
    idx_test = torch.tensor(idx_test, device=device)


    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    test_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    
    labels=torch.LongTensor(labels).to(device)

    train_mask[idx_train] = True
    val_mask[idx_valid] = True
    test_mask[idx_test] = True



    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr , weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=50)

    best_vauc_pr, best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc, final_tauc_pr,best_vauc = 0., 0., 0., 0., 0., 0., 0.,0.,0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy\'s  weight: ', weight)

    time_start = time.time()
    for e in range(1,args.epoch+1):
        model.train()


        nopc=True

        if nopc:
            logits, embeddings = model(features,nopc=True)
            logits = logits.to(device)
            embeddings = embeddings.to(device)

            class_loss = F.cross_entropy(logits[train_mask], torch.tensor(labels[train_mask], device=device),
                                            weight=torch.tensor([1., weight], device=device))

            loss = class_loss
        
        else:
            logits, embeddings, pos_prototype, neg_prototype = model(features)
            logits = logits.to(device)
            embeddings = embeddings.to(device)
            pos_prototype = pos_prototype.to(device)
            neg_prototype = neg_prototype.to(device)

            class_loss = F.cross_entropy(logits[train_mask], torch.tensor(labels[train_mask], device=device),
                                            weight=torch.tensor([1., weight], device=device))

            train_emb = embeddings[train_mask]
            train_labels = torch.tensor(labels[train_mask], device=device)

            pos_idx = (train_labels == 1).nonzero(as_tuple=True)[0]
            neg_idx = (train_labels == 0).nonzero(as_tuple=True)[0]

            min_samples = min(len(pos_idx), len(neg_idx))
            if min_samples > 0:
                if len(pos_idx) > min_samples:
                    pos_idx = pos_idx[torch.randperm(len(pos_idx))[:min_samples]]
                if len(neg_idx) > min_samples:
                    neg_idx = neg_idx[torch.randperm(len(neg_idx))[:min_samples]]

                combined_idx = torch.cat([pos_idx, neg_idx])
                shuffle_idx = torch.randperm(len(combined_idx))
                combined_idx = combined_idx[shuffle_idx]

                selected_emb = train_emb[combined_idx]
                selected_labels = train_labels[combined_idx]

                pos_proto_expanded = pos_prototype.expand(selected_emb.shape[0], -1)
                neg_proto_expanded = neg_prototype.expand(selected_emb.shape[0], -1)

                pos_sim = -F.pairwise_distance(selected_emb, pos_proto_expanded)  # 目标是距离最小化，加上负号，目标是最大化
                neg_sim = -F.pairwise_distance(selected_emb, neg_proto_expanded)
                sim_matrix = torch.cat([neg_sim.unsqueeze(1), pos_sim.unsqueeze(1)], dim=1)

                prototype_loss = F.cross_entropy(sim_matrix, selected_labels)
            else:
                prototype_loss = torch.tensor(0.0, device=train_emb.device, requires_grad=True)

            loss = class_loss + args.alpha * prototype_loss  # 可调整原型损失权重

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()

        logits = logits.detach()
        loss = loss.detach()
        features = features.detach()
        emb = None
    
        with torch.no_grad():
            logits, emb = model(features)
            logits = logits.detach().to(device)
            probs = logits.softmax(1).to(device)

            f1, thres = get_best_f1(labels[val_mask].cpu().numpy(), probs[val_mask].cpu().numpy())
            preds = torch.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1
            trec = recall_score(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy())
            tpre = precision_score(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy())
            tmf1 = f1_score(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), average='macro')
            tauc = roc_auc_score(labels[test_mask].cpu().numpy(), probs[test_mask][:, 1].cpu().numpy())

            precision, recall, _ = precision_recall_curve(labels[test_mask].cpu().numpy(), probs[test_mask][:, 1].cpu().numpy())
            tauc_pr = auc(recall, precision)

            vprecision, vrecall, _ = precision_recall_curve(labels[val_mask].cpu().numpy(), probs[val_mask][:, 1].cpu().numpy())
            vauc_pr = auc(vrecall, vprecision)

            vauc = roc_auc_score(labels[val_mask].cpu().numpy(), probs[val_mask][:, 1].cpu().numpy())

        if (best_vauc + best_vauc_pr + best_f1) < (f1 + vauc + vauc_pr):
            wait_cnt = 0

            best_vauc_pr = vauc_pr
            best_f1 = f1
            best_vauc = vauc
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_tauc_pr = tauc_pr
            print('epoch = ', e, '  Test: recall {:.2f} precision {:.2f} MacroF1 {:.2f} AUC_ROC {:.2f} AUC_PR {:.2f}'.format(final_trec * 100, final_tpre * 100, final_tmf1 * 100, final_tauc * 100, final_tauc_pr * 100))
        else:
            wait_cnt += 1
            if wait_cnt == 400:
                print(f'epoch={e} break!!!')
                break
    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: recall {:.2f} precision {:.2f} MacroF1 {:.2f} AUC_ROC {:.2f}  AUC_PR {:.2f}'.format(final_trec * 100, final_tpre * 100, final_tmf1 * 100, final_tauc * 100, final_tauc_pr * 100))
    print(f'   {final_tmf1 * 100:.2f} | {final_tauc * 100:.2f}| {final_tauc_pr * 100:.2f}    ')
    print(f'   {final_tmf1 * 100:.2f}      {final_tauc * 100:.2f}    {final_tauc_pr * 100:.2f}    ')



    return final_tmf1, final_tauc, final_tauc_pr


def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre