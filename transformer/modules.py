import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class MultiHeadAttention(nn.Module):
    def __innit__(self, d_model, d_k, n_heads):
        super().__init__()
        self.k = nn.Linear(d_model, d_k * n_heads)
        self.q = nn.Linear(d_model, d_k * n_heads)
        self.v = nn.Linear(d_model, d_k * n_heads)
        self.fc = nn.Linear(d_k * n_heads, d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, k, q, v, mask=None):

        # k[N, T, d_k*n_heads]
        K = self.k(k)
        Q = self.q(q)
        V = self.v(v)

        N = K.shape[0]
        T = K.shape[1]

        #k[N, n_heads, T, d_k]
        K = K.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        Q = Q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(N, T, self.n_heads, self.d_k).transpose(1, 2)

        a_score = Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)

        if mask is not None:
            a_score = a_score.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

        a_weights = F.softmax(a_score, dim=-1)

        A = a_weights @ V

        A = A.transpose(1, 2)
        A = A.contiguous().view(N, T, self.d_k * self.n_heads)

        return self.fc(A)
    

class CausalAttention(nn.Module):
    def __innit__(self, d_model, d_k, n_heads, max_len=2048):
        super().__init__()
        self.k = nn.Linear(d_model, d_k * n_heads)
        self.q = nn.Linear(d_model, d_k * n_heads)
        self.v = nn.Linear(d_model, d_k * n_heads)
        self.fc = nn.Linear(d_k * n_heads, d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k

        self.cm = torch.trill(torch.ones(max_len, max_len))
        self.register_buffer('cm', self.cm.view(1, 1 , max_len, max_len))

    def forward(self, k, q, v, pad_mask=None):

        # k[N, T, d_k*n_heads]
        K = self.k(k)
        Q = self.q(q)
        V = self.v(v)

        N = K.shape[0]
        T = K.shape[1]

        #k[N, n_heads, T, d_k]
        K = K.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        Q = Q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(N, T, self.n_heads, self.d_k).transpose(1, 2)

        a_score = Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)

        if pad_mask is not None:
            a_score = a_score.masked_fill(pad_mask[:, None, None, :] == 0, float('-inf'))
            
        a_score = a_score.masked_fill(self.cm[:, :, :T, :T] == 0, float('-inf'))

        a_weights = F.softmax(a_score, dim=-1)

        A = a_weights @ V

        A = A.transpose(1, 2)
        A = A.contiguous().view(N, T, self.d_k * self.n_heads)

        return self.fc(A)
  

class EncoderTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, n_heads, dropout_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, d_k, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(d_model, d_model*4)
        self.fc2 = nn.Linear(d_model*4, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x, mask=None):
        x1 = self.attention(x, x, x, mask=mask)
        x = self.norm1(x + x1)
        x = self.norm1(x)
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x = x + self.dropout2(x1)
        x = self.norm2(x)
        return x
    
    
class DecoderTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, n_heads, max_len, dropout_prob=0.1):
        super().__init__()
        self.attention = CausalAttention(d_model, d_k, n_heads, max_len=max_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(d_model, d_model*4)
        self.fc2 = nn.Linear(d_model*4, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x, pad_mask=None):
        x1 = self.attention(x, x, x, mask=pad_mask)
        x = self.norm1(x + x1)
        x = self.norm1(x)
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x = x + self.dropout2(x1)
        x = self.norm2(x)
        return x

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, d_model, d_k, n_heads, vocab_size, max_len, n_classes, n_layers=4, dropout_prob=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout_prob=dropout_prob)
        transformer_blocks = [EncoderTransformerBlock(d_model, d_k, n_heads, dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        x = self.ln(x)
        x = self.fc(x[:, 0, :])
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_k, d_model, n_heads, n_layers, dropout_prob):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout_prob=dropout_prob)
        transformer_blocks = [DecoderTransformerBlock(d_model, d_k, n_heads, dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)    
    
    def forward(self, x, pad_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, pad_mask=pad_mask)
        x = self.ln(x)
        x = self.fc(x)
        return x








