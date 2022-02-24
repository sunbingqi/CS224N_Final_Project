"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import masked_softmax


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, character_vectors, char_channel_size, char_channel_width, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    character_vectors=character_vectors,
                                    char_channel_size=char_channel_size,
                                    char_channel_width=char_channel_width,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)


        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out



class QANet(nn.Module):
    def __init__(self, word_vectors, character_vectors, hidden_size, char_channel_size, char_channel_width, pad=0, drop_prob=0.1, num_head=1):
        super().__init__()

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    character_vectors=character_vectors,
                                    char_channel_size=char_channel_size,
                                    char_channel_width=char_channel_width,
                                    drop_prob=drop_prob)

        self.num_head = num_head
        self.emb_enc = layers.EncoderBlock(conv_num=4, hidden_size=hidden_size, num_head=num_head, k=7, dropout=0.1)
        self.cq_att = layers.CQAttention(hidden_size=hidden_size)
        self.cq_resizer = layers.Initialized_Conv1d(hidden_size * 4, hidden_size)
        self.model_enc_blks = nn.ModuleList([layers.EncoderBlock(conv_num=2, hidden_size=hidden_size, num_head=num_head, k=5, dropout=0.1) for _ in range(7)])
        self.out = layers.Pointer(hidden_size)
        self.PAD = pad
        self.dropout = drop_prob

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        maskC = torch.zeros_like(cw_idxs) != cw_idxs
        maskQ = torch.zeros_like(qw_idxs) != qw_idxs

        C = self.emb(cw_idxs, cc_idxs)  # (batch_size, c_len, hidden_size)
        Q = self.emb(qw_idxs, qc_idxs)  # (batch_size, q_len, hidden_size)

        Ce = self.emb_enc(C.transpose(1,2), maskC, 1, 1)
        Qe = self.emb_enc(Q.transpose(1,2), maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        log_p1 = masked_softmax(p1.squeeze(), maskC, log_softmax=True)
        log_p2 = masked_softmax(p2.squeeze(), maskC, log_softmax=True)
        return log_p1, log_p2

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters:', params)