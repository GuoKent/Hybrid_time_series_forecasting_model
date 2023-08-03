import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape  # [batch, 8, seq_len, 64]
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # K_expand 增加一个维度然后复制
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # [batch, 8, seq_len, seq_len, 64]
        # 在0-L_K(L_K=96)中随机取数，生成一个L_Q*sample_k的矩阵，即96*25
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q [seq_len, 25]
        # 采样25个key
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # [batch, 8, seq_len, 25, 64]
        # 将Q与采样后的25个key相乘
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # [batch, 8, seq_len, 25]

        # find the Top_k query with sparisty measurement
        # 96个Q中每一个选跟其他K关系最大的值，再算均匀分布差异
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # [batch, 8, seq_len]
        # 对96个评分中选最高的25个，返回值1表示要索引
        M_top = M.topk(n_top, sorted=False)[1]  # [batch, 8, 25]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],  # [batch, 8, 25, 64]
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k  # [batch, 8, 25, 96]

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            # V: [batch, 8, seq_len, 64]
            V_sum = V.mean(dim=-2)  # 未被采样的用V均值替代 V_sum: [batch, 8, 64]
            # 先把96个V都用均值替代
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # [batch, 8, seq_len, 64]
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # [batch, 8, 25, seq_len]
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        # 对25个有Q的更新V，其余均值不变
        # index: [batch, 8, 25]
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        # context_in.shape = [batch, 8, seq_len, 64]

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape  # [batch, seq_len, 8, 64]
        _, L_K, _, _ = keys.shape     # [batch, seq_len, 8, 64]

        # 维度转置
        queries = queries.transpose(2, 1)  # [batch, 8, seq_len, 64]
        keys = keys.transpose(2, 1)        # [batch, 8, seq_len, 64]
        values = values.transpose(2, 1)    # [batch, 8, seq_len, 64]

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)  # 选25个注意力强的
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # ProbAttention中的Prob阶段
        # scores_top: [batch, 8, 25, 96]
        # index: [batch, 8, 25]
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)  # D=64
        if scale is not None:
            scores_top = scores_top * scale  # [batch, 8, 25, 96]
        # get the context  均值替换
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries  更新top25个Q
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        # context: [batch, 8, seq_len, 64]
        # context: [batch, 8, 25, seq_len]

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape  # [batch, seq_len, d_model]
        _, S, _ = keys.shape     # [batch, seq_len, d_model]
        H = self.n_heads         # 8

        # 添加一个维度
        queries = self.query_projection(queries).view(B, L, H, -1)  # [batch, seq_len, 8, 64] 8*64=512
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # self.inner_attention: ProbAttention类  跳转到ProbAttention类的forward
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        # 经过Attention层后经过全连接层 Linear(512, 512)  d_model=512
        return self.out_projection(out), attn
