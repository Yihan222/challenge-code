import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
n_heads = int(config.get('transformer','n_heads'))
batch_size = int(config.get('transformer','batch_size'))
hidden_size = int(config.get('transformer','hidden_size'))
# hidden_size should be 词元维度代表一个词元需要用多长的向量进行表示，词元维度的大小一般是词元的种类数, while rnn use lstm units 60*2 we try the same
src_vocab_size = int(config.get('transformer','src_vocab_size'))
num_hidden_layers = int(config.get('transformer','num_hidden_layers'))
hidden_dropout_prob = float(config.get('transformer','hidden_dropout_prob'))
regressor_dropout_prob = float(config.get('transformer','regressor_dropout_prob'))
d_k = d_v = 64  # K(=Q), V的维度 
max_position_embeddings = int(config.get('transformer','max_position_embeddings'))


def get_attn_pad_mask(seq_q,seq_k):
    batch_size, len_q = seq_q.size() # seq_q 用于升维，为了做attention，mask score矩阵用的
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size,len_q,len_k) # 扩展成多维度   [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):                               # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]          # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #  [batch_size, tgt_len, tgt_len]
    return subsequence_mask

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / hidden_size) for i in range(hidden_size)]
        if pos != 0 else np.zeros(hidden_size) for pos in range(max_position_embeddings)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # even
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # odd
        self.pos_table = torch.FloatTensor(pos_table).cuda()               # enc_inputs: [seq_len, hidden_size]
    
    def forward(self,enc_inputs):                                         # enc_inputs: [batch_size, seq_len, hidden_size]
        enc_inputs += self.pos_table[:enc_inputs.size(1),:]
        return self.dropout(enc_inputs.cuda())

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):                             # Q: [batch_size, n_heads, len_q, d_k]
                                                                       # K: [batch_size, n_heads, len_k, d_k]
                                                                       # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                       # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)                           # 如果是停用词P就等于 0 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(hidden_size, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(hidden_size, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(hidden_size, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, hidden_size, bias=False)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):    # input_Q: [batch_size, len_q, hidden_size]
                                                                # input_K: [batch_size, len_k, hidden_size]
                                                                # input_V: [batch_size, len_v(=len_k), hidden_size]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)              # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)          # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                # [batch_size, len_q, hidden_size]
        return nn.LayerNorm(hidden_size).cuda()(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size*4, hidden_size, bias=False))
        
    def forward(self, inputs):                             # inputs: [batch_size, seq_len, hidden_size]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(hidden_size).cuda()(output + residual)   # [batch_size, seq_len, hidden_size]  

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                                     # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                                        # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):                                # enc_inputs: [batch_size, src_len, hidden_size]
        #输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                          # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,    # enc_outputs: [batch_size, src_len, hidden_size], 
                                               enc_self_attn_mask)                    # attn: [batch_size, n_heads, src_len, src_len]                                                                   
        enc_outputs = self.pos_ffn(enc_outputs)                                       # enc_outputs: [batch_size, src_len, hidden_size]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, hidden_size)
        self.pos_emb = PositionalEncoding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_hidden_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, hidden_size]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, hidden_size]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, hidden_size], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.Regressor = nn.Sequential(
            nn.Dropout(regressor_dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, enc_inputs):
        enc_outputs = self.Encoder(enc_inputs)
        output = self.Regressor(enc_outputs[:, 0, :])  # 取序列的第一个元素作为表征
        return output


 
