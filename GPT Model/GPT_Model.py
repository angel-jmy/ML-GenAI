import torch
import torch.nn as nn
d_k = 64
d_v = 64
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()        
    def forward(self, Q, K, V, attn_mask):        
        # Q K V [batch_size, n_heads, len_q/k/v, dim_q=k/v] (dim_q=dim_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        scores.masked_fill_(attn_mask, -1e9) 
        weights = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(weights, V)
        return context, weights
    
d_embedding = 512
n_heads = 8
batch_size = 3
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, Q, K, V, attn_mask): 
        # Q K V [batch_size,len_q/k/v,embedding_dim]        
        residual, batch_size = Q, Q.size(0)
        # q_s k_s v_s: [batch_size,n_heads.,len_q/k/v,d_q=k/v]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)        
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        # [batch_size,n_heads,len_q,len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        #[batch_size，len_q，n_heads * dim_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) 
        output = self.linear(context)
        #[batch_size, len_q, embedding_dim]
        output = self.layer_norm(output + residual)
        return output, weights

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=d_embedding, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, inputs): 
        # inputs: [batch_size, len_q, embedding_dim]        
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        #[batch_size, len_q, embedding_dim]
        output = self.layer_norm(output + residual)
        return output


import numpy as np
def get_sin_enc_table(n_position, embedding_dim):
    sinusoid_table = np.zeros((n_position, embedding_dim))    
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle    
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    #[batch_size，1，len_k(=len_q)]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    #[batch_size，len_q，len_k]
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k) 
    return pad_attn_mask #[batch_size，len_q，len_k]

def get_attn_subsequent_mask(seq):
    #[batch_size, seq_len(len_q), seq_len(len_k)]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask # [batch_size, seq_len(len_q), seq_len(len_k)]

    
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.feed_forward = PoswiseFeedForwardNet()
        self.norm1 = nn.LayerNorm(d_embedding)
        self.norm2 = nn.LayerNorm(d_embedding)

    def forward(self, dec_inputs, attn_mask=None):
        attn_output, _ = self.self_attn(dec_inputs, dec_inputs, dec_inputs, attn_mask)
        norm1_outputs = self.norm1(dec_inputs + attn_output)
        ff_outputs = self.feed_forward(norm1_outputs)
        dec_outputs = self.norm2(norm1_outputs + ff_outputs)
        return dec_outputs

n_layers = 6
device = "cuda" if torch.cuda.is_available() else "cpu"
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers):
        super(Decoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_embedding)
        self.pos_emb = nn.Embedding(max_seq_len, d_embedding)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs):
        positions = torch.arange(len(dec_inputs), device=dec_inputs.device).unsqueeze(-1)
        inputs_embedding = self.src_emb(dec_inputs) + self.pos_emb(positions)
        attn_mask = get_attn_subsequent_mask(inputs_embedding).to(device)
        for layer in self.layers:
            dec_outputs = layer(inputs_embedding, attn_mask)
        return dec_outputs


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers):
        super(GPT, self).__init__()
        self.decoder = Decoder(vocab_size, max_seq_len, n_layers)
        self.projection = nn.Linear(d_embedding, vocab_size)

    def forward(self, dec_inputs):
        dec_outputs = self.decoder(dec_inputs)
        logits = self.projection(dec_outputs)
        return logits   