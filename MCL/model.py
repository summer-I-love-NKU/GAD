import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class AvgReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class MinReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out

class Contextual_Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Contextual_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1,:], c_mi[:-1,:]),0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits

class Patch_Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Patch_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_ano, h_unano, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_unano, h_ano))
        h_mi = h_ano
        for _ in range(self.negsamp_round):
            h_mi = torch.cat((h_mi[-2:-1, :], h_mi[:-1, :]), 0)
            scs.append(self.f_k(h_unano, h_mi))
        logits = torch.cat(tuple(scs))
        return logits

class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round_patch, negsamp_round_context, readout='avg'):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn_context = GCN(n_in, n_h, activation)
        self.gcn_patch = GCN(n_in, n_h, activation)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.c_disc = Contextual_Discriminator(n_h, negsamp_round_context)
        self.p_disc = Patch_Discriminator(n_h, negsamp_round_patch)

    def forward(self, seq1, adj, sparse=False, msk=None, samp_bias1=None, samp_bias2=None):
        h_1 = self.gcn_context(seq1, adj, sparse)
        h_2 = self.gcn_patch(seq1, adj, sparse)

        if self.read_mode != 'weighted_sum':
            # c：上下文嵌入（通过读出模块聚合得到的图嵌入）。表示取 h_1 的所有节点嵌入，但去掉最后一个节点的嵌入。这里假设最后一个节点是目标节点（h_mv），而其他节点是其上下文。 如果 read_mode == 'avg'，则计算平均值。
            c = self.read(h_1[:, :-1, :])
            # h_mv：目标节点的嵌入（移动节点的嵌入）。 这里假设最后一个节点是目标节点（移动节点），即我们关注的节点。
            h_mv = h_1[:, -1, :]

            # h_unano：非异常节点的嵌入（通常是目标节点的嵌入）。
            # h_ano：异常节点的嵌入（通常是目标节点的前一个节点的嵌入）。
            h_unano = h_2[:, -1, :]#表示取 h_2 的最后一个节点的嵌入。这里假设最后一个节点是非异常节点的嵌入（h_unano）。
            h_ano = h_2[:, -2, :]#表示取 h_2 的倒数第二个节点的嵌入。这里假设倒数第二个节点是异常节点的嵌入（h_ano）。

        else:
            c = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
            h_mv = h_1[:, -1, :]
            h_unano = h_2[:, -1, :]
            h_ano = h_2[:, -2, :]

        ret1 = self.c_disc(c, h_mv, samp_bias1, samp_bias2)
        ret2 = self.p_disc(h_ano, h_unano, samp_bias1, samp_bias2)

        return ret1, ret2


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        '''
        torch.unsqueeze(c, 1)：在第1维（索引为1）插入一个新的维度，将 c 的维度从 [B,F] 变为 [B,1,F]。

        c_x.expand_as(h_pl)：将 c_x 扩展到与 h_pl 相同的维度，即 [B,N,F]。这样，c_x 的每个节点维度都被复制了 N 次。

        维度变化：
        c：[B,F] → [B,1,F] → [B,N,F]

        attn 也就是把全局表示复制了n次，方便后面构造n个正负样本对
        '''
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)  # 原始图 正样本
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)  # 扰动图 负样本
        '''
        self.f_k(h_pl, c_x)：这是一个双线性评分函数，
         ，其中 W 是可学习的权重矩阵。假设 f_k 的输出维度为 [B,N,1]。
        torch.squeeze(..., 2)：移除第2维（索引为2），将输出维度从 [B,N,1] 变为 [B,N]。
        维度变化：  sc_1：[B,N,1] → [B,N]

        '''

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits  # 维度是  batchsize * 2n


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        # attn c是全局表示，用的平均值,1*d维度。   h1和h2分别是原始图和扰动图的节点表示：n*d维度
        c = self.read(h_1)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret