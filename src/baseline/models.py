import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GAMENet(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0'), ddi_in_memory=True):
        super(GAMENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=0.4)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K + 1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 8, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

    def forward(self, input):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        i3_seq = []
        i4_seq = []

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(self.dropout(
                self.embeddings[0](torch.LongTensor(adm[9]).unsqueeze(dim=0).to(self.device))))  # (1,1,dim)
            i2 = mean_embedding(
                self.dropout(self.embeddings[1](torch.LongTensor(adm[10]).unsqueeze(dim=0).to(self.device))))
            i3 = mean_embedding(
                self.dropout(self.embeddings[2](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i4 = mean_embedding(
                self.dropout(self.embeddings[2](torch.LongTensor(adm[5]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
            i3_seq.append(i3)
            i4_seq.append(i4)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)
        i3_seq = torch.cat(i3_seq, dim=1)
        i4_seq = torch.cat(i4_seq, dim=1)

        o1, h1 = self.encoders[0](
            i1_seq
        )  # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        o2, h2 = self.encoders[1](
            i2_seq
        )
        o3, h3 = self.encoders[2](
            i3_seq
        )
        o4, h4 = self.encoders[3](
            i4_seq
        )
        patient_representations = torch.cat([o1, o2, o3, o4], dim=-1).squeeze(dim=0)  # (seq, dim*8)
        queries = self.query(patient_representations)  # (seq, dim)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:]  # (1,dim)

        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()

        if len(input) > 1:
            history_keys = queries[:(queries.size(0) - 1)]  # (seq-1, dim)

            history_values = np.zeros((len(input) - 1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input) - 1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device)  # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()))  # (1, seq-1)
            weighted_values = visit_weight.mm(history_values)  # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory)  # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1))  # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)


class Leap(nn.Module):
    def __init__(self, voc_size, emb_dim=128, device=torch.device('cpu:0')):
        super(Leap, self).__init__()
        self.voc_size = voc_size
        self.device = device
        self.SOS_TOKEN = voc_size[2]
        self.END_TOKEN = voc_size[2] + 1

        self.enc_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], emb_dim, ),
            nn.Dropout(0.3)
        )

        self.dec_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 2, emb_dim, ),
            nn.Dropout(0.3)
        )

        self.dec_gru = nn.GRU(emb_dim * 2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim * 2, 1)

        self.output = nn.Linear(emb_dim, voc_size[2] + 2)

    def forward(self, input, max_len=20):
        device = self.device
        # input (3, codes)
        input_diag_tensor = torch.LongTensor(input[9]).to(device)
        # (len, dim)
        input_embedding = self.enc_embedding(input_diag_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        output_logits = []
        hidden_state = None
        if self.training:
            for med_code in [self.SOS_TOKEN] + input[0]:
                dec_input = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0)  # (1,dim)

                if hidden_state is None:
                    hidden_state = dec_input

                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1)  # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1)  # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)  # (1, len)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)

                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
                                               hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)

                output_logits.append(self.output(F.relu(hidden_state)))

            return torch.cat(output_logits, dim=0)

        else:
            for di in range(max_len):
                if di == 0:
                    dec_input = torch.LongTensor([[self.SOS_TOKEN]]).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0)  # (1,dim)
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1)  # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1)  # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)  # (1, len)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)
                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
                                               hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)
                output = self.output(F.relu(hidden_state))
                topv, topi = output.data.topk(1)
                output_logits.append(F.softmax(output, dim=-1))
                dec_input = topi.detach()
            return torch.cat(output_logits, dim=0)


class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2] * 3
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.3)
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, input):
        device = self.device
        # input: (visit, 3, codes )
        max_len = max([(len(v[9]) + len(v[10]) + len(v[1]) + len(v[5]) + len(v[0])) for v in input])
        input_np = []
        for visit in input:
            input_tmp = []
            input_tmp.extend(visit[9])
            input_tmp.extend(list(np.array(visit[10]) + self.voc_size[0]))
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0] + self.voc_size[1]))
            input_tmp.extend(list(np.array(visit[5]) + self.voc_size[0] + self.voc_size[1] + self.voc_size[2]))
            input_tmp.extend(list(np.array(visit[0]) + self.voc_size[0] + self.voc_size[1] + self.voc_size[2] * 2))
            if len(input_tmp) < max_len:
                input_tmp.extend( [self.input_len]*(max_len - len(input_tmp)) )

            input_np.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(input_np).to(device)) # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1) # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0)) # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0)) # h: (1, visit, emb)

        g = g.squeeze(dim=0) # (visit, emb)
        h = h.squeeze(dim=0) # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1) # (visit, 1)
        attn_h = F.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        return self.output(c)


class MICRON(nn.Module):
    def __init__(self, vocab_size, ddi_adj, emb_dim=256, device=torch.device('cpu:0')):
        super(MICRON, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(3)])
        self.dropout = nn.Dropout(p=0.5)

        self.health_net = nn.Sequential(
            nn.Linear(4 * emb_dim, emb_dim)
        )

        #
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, vocab_size[2])
        )

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.init_weights()

    def forward(self, input):

        # patient health representation
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        diag_emb = sum_embedding(self.dropout(
            self.embeddings[0](torch.LongTensor(input[-1][9]).unsqueeze(dim=0).to(self.device))))  # (1,1,dim)
        prod_emb = sum_embedding(self.dropout(
            self.embeddings[1](torch.LongTensor(input[-1][10]).unsqueeze(dim=0).to(self.device))))
        adm_emb = sum_embedding(self.dropout(
            self.embeddings[2](torch.LongTensor(input[-1][1]).unsqueeze(dim=0).to(self.device))))
        pre_emb = sum_embedding(self.dropout(
            self.embeddings[2](torch.LongTensor(input[-1][5]).unsqueeze(dim=0).to(self.device))))

        # diag_emb = torch.cat(diag_emb, dim=1) #(1,seq,dim)
        # prod_emb = torch.cat(prod_emb, dim=1) #(1,seq,dim)

        if len(input) < 2:
            diag_emb_last = diag_emb * torch.tensor(0.0)
            prod_emb_last = diag_emb * torch.tensor(0.0)
            adm_emb_last = diag_emb * torch.tensor(0.0)
            pre_emb_last = diag_emb * torch.tensor(0.0)
        else:
            diag_emb_last = sum_embedding(self.dropout(
                self.embeddings[0](torch.LongTensor(input[-2][0]).unsqueeze(dim=0).to(self.device))))  # (1,1,dim)
            prod_emb_last = sum_embedding(self.dropout(
                self.embeddings[1](torch.LongTensor(input[-2][1]).unsqueeze(dim=0).to(self.device))))
            adm_emb_last = sum_embedding(self.dropout(
                self.embeddings[2](torch.LongTensor(input[-2][1]).unsqueeze(dim=0).to(self.device))))
            pre_emb_last = sum_embedding(self.dropout(
                self.embeddings[2](torch.LongTensor(input[-2][5]).unsqueeze(dim=0).to(self.device))))
            # diag_emb_last = torch.cat(diag_emb_last, dim=1) #(1,seq,dim)
            # prod_emb_last = torch.cat(prod_emb_last, dim=1) #(1,seq,dim)

        health_representation = torch.cat([diag_emb, prod_emb, adm_emb, pre_emb], dim=-1).squeeze(dim=0)  # (seq, dim*4)
        health_representation_last = torch.cat([diag_emb_last, prod_emb_last, adm_emb_last, pre_emb_last], dim=-1).squeeze(dim=0)  # (seq, dim*4)

        health_rep = self.health_net(health_representation)[-1:, :]  # (seq, dim)
        health_rep_last = self.health_net(health_representation_last)[-1:, :]  # (seq, dim)
        health_residual_rep = health_rep - health_rep_last

        # drug representation
        drug_rep = self.prescription_net(health_rep)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)

        # reconstructon loss
        rec_loss = 1 / self.tensor_ddi_adj.shape[0] * torch.sum(
            torch.pow((F.sigmoid(drug_rep) - F.sigmoid(drug_rep_last + drug_residual_rep)), 2))

        # ddi_loss
        neg_pred_prob = F.sigmoid(drug_rep)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 1 / self.tensor_ddi_adj.shape[0] * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
