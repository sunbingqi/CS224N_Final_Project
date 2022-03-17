"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class EmbeddingBASE(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.
    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(EmbeddingBASE, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class Embedding(nn.Module):

    def __init__(self, word_vectors, hidden_size, character_vectors, char_channel_size, char_channel_width, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed_layer = CharEmbedding(hidden_size, character_vectors, character_vectors.size(1), char_channel_size, char_channel_width, drop_prob)
        self.proj = nn.Linear(word_vectors.size(1) + char_channel_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x1, x2):
        word_emb = self.word_embed(x1)   # (batch_size, seq_len, embed_size)
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        char_emb = self.char_embed_layer(x2)

        emb = torch.cat([word_emb, char_emb], dim=-1)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class CharEmbedding(nn.Module):

    def __init__(self, hidden_size, character_vectors, char_dim, char_channel_size, char_channel_width, drop_prob):
        super(CharEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.char_embed = nn.Embedding.from_pretrained(character_vectors, freeze=False)
        self.char_dim = char_dim
        self.char_channel_size = char_channel_size
        self.char_conv = nn.Conv2d(1, char_channel_size, (char_dim, char_channel_width))

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.dropout(self.char_embed(x), self.drop_prob, self.training)
        x = x.transpose(2, 3)
        x = x.view(-1, self.char_dim, x.size(3))
        x = x.unsqueeze(1)
        x = self.char_conv(x)
        x = F.relu(x)
        x = x.squeeze()
        x = F.max_pool1d(x, x.size(2)).squeeze()
        out = x.view(batch_size, -1, self.char_channel_size)
        return out


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers,
        #                   batch_first=True,
        #                   bidirectional=True,
        #                   dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class EncoderBlock(nn.Module):
    def __init__(self, conv_num, hidden_size, num_head, k, drop_prob=0.1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(hidden_size, hidden_size, k) for _ in range(conv_num)])
        self.conv_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(conv_num)])
        self.self_att = SelfAttention(hidden_size, num_head, drop_prob=drop_prob)
        self.ffd_1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.ffd_1.weight, nonlinearity='relu')
        self.ffd_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.ffd_2.weight)
        self.norm_att = nn.LayerNorm(hidden_size)
        self.norm_ffd = nn.LayerNorm(hidden_size)
        self.conv_num = conv_num
        self.drop_prob = drop_prob

    def forward(self, x, mask, layer, num_blks):
        total_layers = (self.conv_num + 1) * num_blks
        out = self.enc_pos(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.conv_norms[i](out.permute((0, 2, 1))).permute((0, 2, 1))
            out = F.dropout(out, p=self.drop_prob, training=self.training)
            out = conv(out)
            if self.training:
                out = self.layer_dropout(out, res, self.drop_prob * float(layer)/total_layers)
            layer += 1
        res = out
        
        out = self.norm_att(out.permute((0, 2, 1))).permute((0, 2, 1))
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.self_att(out, mask)
        if self.training:
            out = self.layer_dropout(out, res, self.drop_prob * float(layer)/total_layers)
        layer += 1
        res = out

        out = self.norm_ffd(out.permute((0, 2, 1))).permute((0, 2, 1))
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.ffd_1(out)
        out = F.relu(out)
        out = self.ffd_2(out)
        if self.training:
            out = self.layer_dropout(out, res, self.drop_prob * float(layer)/total_layers)
        return out

    # reference: https://github.com/XinyiLiu0227/NLP/blob/007cf2a30ed2dd983ddad9b36afcdd3d808db0b8/QA_code/encoder.py#L134
    def enc_pos(self, x, min_timescale=1.0, max_timescale=1.0e4):
        x = x.transpose(1, 2)
        length = x.shape[1]
        channels = x.shape[2]
        signal = self.get_timing_signal(length, channels, min_timescale, max_timescale)
        return (x + signal.to(x.get_device())).transpose(1, 2)

    def get_timing_signal(self, length, channels, min_timescale=1.0, max_timescale=1.0e4):
        position = torch.arange(length).type(torch.float32)
        num_timescales = channels // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
        signal = m(signal)
        signal = signal.view(1, length, channels)
        return signal

    def layer_dropout(self, inputs, residual, drop_prob):
        return residual if random.uniform(0, 1) < drop_prob else inputs + residual


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.depth_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.point_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=True)
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = F.relu(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_head, drop_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.drop_prob = drop_prob
        self.key_value_conv = nn.Conv1d(hidden_size, hidden_size*2, kernel_size=1)
        nn.init.xavier_uniform_(self.key_value_conv.weight)
        self.query_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        nn.init.xavier_uniform_(self.query_conv.weight)

    def forward(self, x, mask):
        key_value = self.key_value_conv(x)
        query = self.query_conv(x)
        key_value = key_value.permute((0, 2, 1))
        query = query.permute((0, 2, 1))

        K, V = [self.split_last_dim(kv, self.num_head) for kv in torch.split(key_value, self.hidden_size, dim=2)]
        Q = self.split_last_dim(query, self.num_head)

        key_depth_per_head = self.hidden_size // self.num_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).permute((0, 2, 1))

    def dot_product_attention(self, q, k, v, mask):
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        shapes = [x if x != None else -1 for x in list(logits.size())]
        mask = mask.view(shapes[0], 1, 1, shapes[-1])
        logits = logits * mask + (1 - mask.float()) * (-1e30)
        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.drop_prob, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class QANetOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = nn.Conv1d(hidden_size*2, 1, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.w1.weight)
        self.w2 = nn.Conv1d(hidden_size * 2, 1, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, mod_1, mod_2, mod_3, mask):
        x_1 = torch.cat([mod_1, mod_2], dim=1)
        x_2 = torch.cat([mod_1, mod_3], dim=1)
        x_1 = self.w1(x_1)
        x_2 = self.w1(x_2)
        log_p1 = masked_softmax(x_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(x_2.squeeze(), mask, log_softmax=True)
        return log_p1, log_p2


