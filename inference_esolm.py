import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
from dataclasses import dataclass

# --- Configuration (Must match train_esolm.py) ---
@dataclass
class Config:
    vocab_size = 8000
    seq_len = 128
    n_layers = 12
    n_heads = 12
    dim = 768
    dropout = 0.05
    alpha0 = 0.5
    mask_token_id = 0

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Components (Must match train_esolm.py) ---
def get_causal_mask(T, device):
    mask = torch.tril(torch.ones(T, T, device=device)).bool()
    attn_bias = torch.zeros(T, T, device=device).masked_fill(~mask, float('-inf'))
    return attn_bias.unsqueeze(0).unsqueeze(0)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.head_dim)
    def forward(self, x, attn_bias=None):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(L)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None: att = att + attn_bias.to(q.dtype)
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
        self.attn = Attention()
        self.ff = nn.Sequential(
            nn.Linear(cfg.dim, 4 * cfg.dim), nn.GELU(),
            nn.Linear(4 * cfg.dim, cfg.dim), nn.Dropout(cfg.dropout)
        )
    def forward(self, x, attn_bias=None):
        x = x + self.attn(self.ln1(x), attn_bias=attn_bias)
        x = x + self.ff(self.ln2(x))
        return x

class EsoLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size)
    def forward(self, x, attn_bias=None):
        h = self.token_emb(x)
        for block in self.blocks: h = block(h, attn_bias=attn_bias)
        return self.head(self.ln(h))

def get_esolm_b_diffusion_mask(is_masked):
    B, T = is_masked.shape
    device = is_masked.device
    rand_keys = torch.rand(B, T, device=device)
    sort_keys = rand_keys + is_masked.float()
    sigma_inv = torch.argsort(torch.argsort(sort_keys, dim=1), dim=1)
    allowed = sigma_inv.unsqueeze(2) >= sigma_inv.unsqueeze(1)
    attn_bias = torch.zeros(B, T, T, device=device).masked_fill_(~allowed, float('-inf'))
    return attn_bias.unsqueeze(1)

@torch.no_grad()
def generate(model, sp, prompt, steps=16, temperature=0.8):
    model.eval()
    tokens = sp.encode(prompt)
    x = torch.tensor([tokens], device=device)
    B, L0 = x.shape
    gen_len = cfg.seq_len - L0
    x = torch.cat([x, torch.full((1, gen_len), cfg.mask_token_id, device=device)], dim=1)
    
    print(f"Phase 1: Diffusion ({steps} steps)...")
    for step in range(steps):
        is_masked = (x == cfg.mask_token_id)
        if not is_masked.any(): break
        
        # Eso-LM B mask for diffusion
        attn_bias = get_esolm_b_diffusion_mask(is_masked)
        logits = model(x, attn_bias=attn_bias)
        
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.distributions.Categorical(probs).sample()
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)
            sampled = torch.argmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values

        x[is_masked] = sampled[is_masked]
        
        # Stop diffusion once we reach alpha0 threshold
        # Or just use the scheduled steps
        remaining_ratio = 1 - (step + 1) / steps
        k = int(remaining_ratio * gen_len)
        if k > 0:
            masked_conf = confidence.clone()
            masked_conf[~is_masked] = 999
            idx = torch.topk(masked_conf.view(-1), k, largest=False).indices
            x.view(-1)[idx] = cfg.mask_token_id

    # Phase 2: Sequential AR Fill
    is_masked = (x == cfg.mask_token_id)
    if is_masked.any():
        print("Phase 2: Sequential AR Refinement...")
        # We fill remaining masks one by one, left-to-right
        mask_indices = torch.where(is_masked[0])[0]
        for idx in mask_indices:
            # For AR phase, we can use causal mask or just let it see all previously generated tokens
            # Paper suggests causal attention for the AR phase
            causal_bias = get_causal_mask(cfg.seq_len, device)
            logits = model(x, attn_bias=causal_bias)
            
            if temperature > 0:
                probs = F.softmax(logits[0, idx] / temperature, dim=-1)
                token = torch.distributions.Categorical(probs).sample()
            else:
                token = torch.argmax(logits[0, idx])
            
            x[0, idx] = token
            
    return sp.decode(x[0].tolist())

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor()
    sp.load('tinystories_bpe.model')
    
    model = EsoLM().to(device)
    model.load_state_dict(torch.load("esolm_tinystories.pt", weights_only=True))
    print("Model loaded successfully.")
    
    while True:
        prompt = input("\nEnter prompt (or 'q' to quit): ")
        if prompt.lower() == 'q': break
        result = generate(model, sp, prompt)
        print("-" * 30)
        print(result)
        print("-" * 30)
