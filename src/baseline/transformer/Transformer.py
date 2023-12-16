import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Transformer(nn.Module):
    def __init__(self, voc_size, emb_dim=64, feature_size=8, device=torch.device('cpu:0')):
        super(Transformer, self).__init__()
        self.emb_dim = emb_dim
        self.feature_size = feature_size
        self.device = device
        self.nhead = 2
        self.voc_size = voc_size
        self.SOS_TOKEN = voc_size[2]  # start of sentence
        self.END_TOKEN = voc_size[2] + 1  # end   新增的两个编码，两者均是针对于药物的embedding
        self.MED_PAD_TOKEN = voc_size[2] + 2  # 用于embedding矩阵中的padding（全为0）
        self.DIAG_PAD_TOKEN = voc_size[0] + 2
        self.PROC_PAD_TOKEN = voc_size[1] + 2

        self.diag_embedding = nn.Sequential(nn.Embedding(voc_size[0] + 3, self.emb_dim, self.DIAG_PAD_TOKEN), nn.Dropout(0.3))
        self.proc_embedding = nn.Sequential(nn.Embedding(voc_size[1] + 3, self.emb_dim, self.PROC_PAD_TOKEN), nn.Dropout(0.3))
        self.med_embedding = nn.Sequential(nn.Embedding(voc_size[2] + 3, self.emb_dim, self.MED_PAD_TOKEN), nn.Dropout(0.3))

        self.diag_encoder = nn.TransformerEncoderLayer(self.emb_dim, self.nhead, batch_first=True, dropout=0.2)
        self.proc_encoder = nn.TransformerEncoderLayer(self.emb_dim, self.nhead, batch_first=True, dropout=0.2)
        self.adm_encoder = nn.TransformerEncoderLayer(self.emb_dim, self.nhead, batch_first=True, dropout=0.2)
        self.pre_encoder = nn.TransformerEncoderLayer(self.emb_dim, self.nhead, batch_first=True, dropout=0.2)
        self.last_dis_encoder = nn.TransformerEncoderLayer(self.emb_dim, self.nhead, batch_first=True, dropout=0.2)

        self.decoder = MedTransformerDecoder(self.emb_dim, self.nhead, dim_feedforward=emb_dim * 2, dropout=0.2, layer_norm_eps=1e-5)

        self.get_g_score = nn.Linear(self.emb_dim, voc_size[2] + 2)

    def encode(self, diag, proc, dis_med, adm_med, pre_med, diag_mask, proc_mask, dis_mask, adm_mask, pre_mask):
        batch_size, max_visit_num, med_max = dis_med.size()
        diag_max = diag.size(2)
        proc_max = proc.size(2)

        input_diag_embedding = self.diag_embedding(diag).view(batch_size * max_visit_num, diag_max, self.emb_dim)
        input_proc_embedding = self.diag_embedding(proc).view(batch_size * max_visit_num, proc_max, self.emb_dim)
        diag_enc_mask = diag_mask.view(batch_size * max_visit_num, diag_max).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, diag_max, 1)
        diag_enc_mask = diag_enc_mask.view(batch_size * max_visit_num * self.nhead, diag_max, diag_max)
        proc_enc_mask = proc_mask.view(batch_size * max_visit_num, proc_max).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, proc_max, 1)
        proc_enc_mask = proc_enc_mask.view(batch_size * max_visit_num * self.nhead, proc_max, proc_max)
        input_diag_embedding = self.diag_encoder(input_diag_embedding, src_mask=diag_enc_mask).view(batch_size, max_visit_num, diag_max, self.emb_dim)
        input_proc_embedding = self.proc_encoder(input_proc_embedding, src_mask=proc_enc_mask).view(batch_size, max_visit_num, proc_max, self.emb_dim)

        input_adm_embedding = self.med_embedding(adm_med).view(batch_size * max_visit_num, med_max, self.emb_dim)
        input_pre_embedding = self.med_embedding(pre_med).view(batch_size * max_visit_num, med_max, self.emb_dim)
        adm_enc_mask = adm_mask.view(batch_size * max_visit_num, med_max).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, med_max, 1)
        adm_enc_mask = adm_enc_mask.view(batch_size * self.nhead * max_visit_num, med_max, med_max)
        pre_enc_mask = pre_mask.view(batch_size * max_visit_num, med_max).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, med_max, 1)
        pre_enc_mask = pre_enc_mask.view(batch_size * self.nhead * max_visit_num, med_max, med_max)

        # 区分入院药物与住院药物

        input_adm_embedding = self.adm_encoder(input_adm_embedding, src_mask=adm_enc_mask).view(batch_size, max_visit_num, med_max, self.emb_dim)
        input_pre_embedding = self.pre_encoder(input_pre_embedding, src_mask=pre_enc_mask).view(batch_size, max_visit_num, med_max, self.emb_dim)

        return input_diag_embedding, input_proc_embedding, input_adm_embedding, input_pre_embedding

    def decode(self, input_medications, input_med_mask, input_disease_embedding, input_proc_embedding, input_adm_embedding, input_pre_embedding, diag_mask, proc_mask, adm_mask, pre_mask):
        batch_size, max_visit_num, med_max = input_medications.size()
        diag_max = input_disease_embedding.size(2)
        proc_max = input_proc_embedding.size(2)
        med_max_1 = input_adm_embedding.size(2)
        input_med_embedding = self.med_embedding(input_medications).view(batch_size * max_visit_num, med_max, self.emb_dim)
        input_m_enc_mask = input_med_mask.view(batch_size * max_visit_num, med_max).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, med_max, 1)
        med_self_mask = input_m_enc_mask.view(batch_size * max_visit_num * self.nhead, med_max, med_max)
        diag_att_mask = diag_mask.view(batch_size * max_visit_num, diag_max).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, med_max, 1)
        diag_att_mask = diag_att_mask.view(batch_size * max_visit_num * self.nhead, med_max, diag_max)
        proc_att_mask = proc_mask.view(batch_size * max_visit_num, proc_max).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, med_max, 1)
        proc_att_mask = proc_att_mask.view(batch_size * max_visit_num * self.nhead, med_max, proc_max)
        adm_att_mask = adm_mask.view(batch_size * max_visit_num, med_max_1).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, med_max, 1)
        adm_att_mask = adm_att_mask.view(batch_size * max_visit_num * self.nhead, med_max, med_max_1)
        pre_att_mask = pre_mask.view(batch_size * max_visit_num, med_max_1).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, med_max, 1)
        pre_att_mask = pre_att_mask.view(batch_size * max_visit_num * self.nhead, med_max, med_max_1)

        dec_hidden = self.decoder(input_medication_embedding=input_med_embedding,
                                  input_disease_embedding=input_disease_embedding.view(batch_size * max_visit_num, diag_max, -1),
                                  input_proc_embedding=input_proc_embedding.view(batch_size * max_visit_num, proc_max, -1),
                                  input_adm_embedding=input_adm_embedding.view(batch_size * max_visit_num, med_max_1, -1),
                                  input_pre_embedding=input_pre_embedding.view(batch_size * max_visit_num, med_max_1, -1),
                                  input_medication_self_mask=med_self_mask, d_mask=diag_att_mask, p_mask=proc_att_mask,
                                  adm_mask=adm_att_mask, pre_mask=pre_att_mask)

        score_g = self.get_g_score(dec_hidden)
        score_g = score_g.view(batch_size, max_visit_num, med_max, -1)
        prob_g = F.softmax(score_g, dim=-1)
        return torch.log(prob_g)

    def forward(self, dis_med, adm_med, pre_med, diag, proc, dis_mask, adm_mask, pre_mask, diag_mask, proc_mask):
        batch_size, max_visit_num, med_max = dis_med.size()
        input_diag_embedding, input_proc_embedding, input_adm_embedding, input_pre_embedding = self.encode(diag, proc,
            dis_med, adm_med, pre_med, diag_mask, proc_mask, dis_mask, adm_mask, pre_mask)

        input_med = torch.full((batch_size, max_visit_num, 1), self.SOS_TOKEN).to(self.device)
        input_med = torch.cat([input_med, dis_med], dim=2)
        dis_sos_mask = torch.zeros((batch_size, max_visit_num, 1), device=self.device).float()
        input_med_mask = torch.cat([dis_sos_mask, dis_mask], dim=2)
        output_prob = self.decode(input_med, input_med_mask, input_diag_embedding, input_proc_embedding, input_adm_embedding,
                                  input_pre_embedding, diag_mask, proc_mask, adm_mask, pre_mask)

        return output_prob


class MedTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5) -> None:
        super(MedTransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2p_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2ma_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2mp_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.nhead = nhead

    def forward(self, input_medication_embedding, input_disease_embedding, input_proc_embedding, input_adm_embedding, input_pre_embedding,
                 input_medication_self_mask,  d_mask, p_mask, adm_mask, pre_mask):

        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            input_medication_embedding: [*, max_med_num+1, embedding_size]
        Shape:
            see the docs in Transformer class.
        """
        input_len = input_medication_embedding.size(0)
        tgt_len = input_medication_embedding.size(1)
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len, input_len * self.nhead, input_disease_embedding.device)
        self_attn_mask = subsequent_mask + input_medication_self_mask

        x = input_medication_embedding
        x = self.norm1(x + self._sa_block(x, self_attn_mask))
        x = self.norm2( x + self._m2d_mha_block(x, input_disease_embedding, d_mask) + self._m2p_mha_block(x, input_proc_embedding, p_mask)
                        + self._m2ma_mha_block(x, input_adm_embedding, adm_mask) + self._m2p_mha_block(x, input_pre_embedding, pre_mask))
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _m2d_mha_block(self, x, mem, attn_mask):
        x = self.m2d_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _m2p_mha_block(self, x, mem, attn_mask):
        x = self.m2p_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _m2ma_mha_block(self, x, mem, attn_mask):
        x = self.m2ma_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _m2mp_mha_block(self, x, mem, attn_mask):
        x = self.m2mp_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def generate_square_subsequent_mask(self, sz: int, batch_size: int, device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        return mask


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GCN_dmo(nn.Module):
    def __init__(self, diag_voc_size, med_voc_size, emb_dim, dmo_adj, device=torch.device('cpu:0')):
        super(GCN_dmo, self).__init__()
        self.diag_voc_size = diag_voc_size
        self.med_voc_size = med_voc_size
        self.emb_dim = emb_dim
        self.dmo_adj = dmo_adj
        self.device = device

        self.adjust = nn.Linear(self.med_voc_size, self.diag_voc_size)
        self.x = torch.eye(diag_voc_size).to(device)
        self.gcn1 = GraphConvolution(diag_voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        dmo_adj = torch.FloatTensor(self.normalize(self.dmo_adj)).to(self.device)
        dmo_adjust = self.adjust(dmo_adj) + self.x
        dmo_node_embedding = self.gcn1(self.x, dmo_adjust)
        dmo_node_embedding = self.dropout(dmo_node_embedding)
        dmo_node_embedding = self.gcn2(dmo_node_embedding, dmo_adjust)
        return dmo_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
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
