import torch
import torch.nn as nn

# =========================
# rotary embeddings
# =========================

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_head, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

def apply_rotary(X, cos, sin):
    # X: (batch_size, n_heads, seq_len, d_head)
    d_half = X.shape[-1] // 2
    x1, x2 = X[..., :d_half], X[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

# =========================
# layers
# =========================

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, X):
        return self.W2(torch.relu(self.W1(X)))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(self.d_head, max_seq_len)

    def forward(self, X):
        batch_size, seq_len, d_model = X.shape
        Q = self.Q(X).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.K(X).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.V(X).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(seq_len, X.device)
        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, X):
        X = X + self.attn(self.ln1(X))
        X = X + self.ff(self.ln2(X))
        return X

# =========================
# full model
# =========================

# K⊗L looped transformer
class Transformer(nn.Module):
    def __init__(self, K, L, vocab_size, d_model, n_heads, d_ff, max_seq_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, max_seq_len) for _ in range(K)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.K = K
        self.L = L

    def forward(self, X):
        batch_size, seq_len = X.shape
        X = self.token_embeddings(X)
        for _ in range(self.L):
            for block in self.blocks:
                X = block(X)
        X = self.ln_final(X)
        return self.lm_head(X)