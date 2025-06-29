import sympy
import scipy
from torch.nn import init
import torch.nn.functional as F
import dgl.function as fn
import torch
from dgl.nn.pytorch.factory import KNNGraph
import dgl
import torch.nn as nn
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu, bias=True):
        super(GCN, self).__init__()
        self.conv = GraphConv(in_feats, out_feats, norm='both', weight=True, bias=bias)
        self.activation = activation

    def forward(self, graph, feat):
        h = self.conv(graph, feat)
        if self.activation is not None:
            h = self.activation(h)
        return h
    

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        # self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

        self.learnable_diag = torch.nn.Parameter(torch.rand(3,in_feats))

        self.linear = nn.Linear(out_feats, out_feats, True)
        self.reset_parameters()


    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph, learnable_diag):
            graph.ndata['h'] = feat * D_invsqrt
            # graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h'))
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))###########################attn

            return feat - torch.mul( (graph.ndata.pop('h') * D_invsqrt) , learnable_diag)

        # def unnLaplacian(feat, D_invsqrt, graph):
        #     n = graph.number_of_nodes()
        #     adj = graph.adj(transpose=True, scipy_fmt=graph.formats()['created'][0]).astype(float)
        #     norm = sparse.diags(Fb.asnumpy(graph.in_degrees()).clip(1) ** -0.5, dtype=float)
        #     laplacian = sparse.eye(n) - norm * adj * norm
        #     laplacian = torch.from_numpy(laplacian.todense())
        #     return torch.mm(laplacian.float(),feat)

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1)
            h = self._theta[0]*feat
            h = torch.mul(h , self.learnable_diag[0])
            h = self.linear(h)
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph,self.learnable_diag[k])
                h += self._theta[k]*feat
        return h

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas



class ComponentAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(ComponentAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        # print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        # print(beta)   # debug
        return (beta * z).sum(1)  # (N, D * K)

class MGEPC(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, knng, args, d=2, batch=False):
        super(MGEPC, self).__init__()
        self.comp_att = ComponentAttention(in_size=h_feats,hidden_size=h_feats)
        self.g = graph
        self.args = args
        self.knng = knng
        thetas_val = calculate_theta2(d=d)
        self.thetas = thetas_val


        self.conv = []
        self.conv1 = PolyConv(h_feats, h_feats, self.thetas[0], lin=False)
        self.conv2 = PolyConv(h_feats, h_feats, self.thetas[1], lin=False)
        self.conv3 = PolyConv(h_feats, h_feats, self.thetas[2], lin=False)
        self.conv = [self.conv1,self.conv2,self.conv3]

        self.conv_knn = []
        self.convk1 = PolyConv(h_feats, h_feats, [1.0]*3, lin=False)
        self.convk2 = PolyConv(h_feats, h_feats, [1.0]*3, lin=False)
        self.convk3 = PolyConv(h_feats, h_feats, [1.0]*3, lin=False)
        self.conv_knn = [self.convk1,self.convk2,self.convk3]

        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)

        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)

        self.linear_knn = nn.Linear(h_feats*len(self.conv_knn), h_feats)

        self.linear4 = nn.Linear(h_feats,num_classes)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(args.drop_ratio)

        self.d = d

    def forward(self, in_feat, nopc=False):
        ##########################################
        # h
        h = self.linear(in_feat)
        h = self.drop(h)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h = self.drop(h)

        ##########################################
        # h_o
        h_final = torch.zeros([len(in_feat), 0]).to(in_feat.device)#################
        for conv in self.conv:
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)

        h_final = self.drop(h_final)
        h_o = self.linear3(h_final)
        h_o = self.act(h_o)

        ##########################################
        # h_knn
        h_knn = torch.zeros([len(in_feat), 0]).to(in_feat.device)
        for conv in self.conv_knn:
            hk0 = conv(self.knng, h)
            # hk0 = 3*h - conv(self.knng, h)
            h_knn = torch.cat([h_knn, hk0], -1)

        h_knn = self.drop(h_knn)
        h_knn = self.linear_knn(h_knn)
        h_knn = self.act(h_knn)
        ##########################################
        # 节点表示：emb
        z = torch.stack([h_o,h_knn],dim=1)# (n,2,64)
        h = self.comp_att(z) # (n,64)
        emb = h
        ##########################################
        # 输出的logit：h
        h = self.linear4(h) # (n,2)
        ##########################################

        if self.training and not nopc:
            train_mask = self.g.ndata['train_mask'].bool()
            train_emb = emb[train_mask]
            train_label = self.g.ndata['label'][train_mask]
            
            pos_mask = train_label == 1
            neg_mask = train_label == 0
            
            # 计算正负原型
            pos_prototype = torch.mean(train_emb[pos_mask], dim=0, keepdim=True) if pos_mask.any() else None
            neg_prototype = torch.mean(train_emb[neg_mask], dim=0, keepdim=True) if neg_mask.any() else None
            
            return h, emb, pos_prototype, neg_prototype
        else:
            return h, emb