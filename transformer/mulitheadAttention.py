import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __innit__(self, d_model, d_k, n_heads):
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







