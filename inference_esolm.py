import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
import numpy as np
import os

# --- Configuration (Must match train_esolm.py) ---
class Config:
    vocab_size = 10000
    seq_len = 256
    n_layers = 8
    n_heads = 8
    dim = 512
    dropout = 0.1
    mask_token_id = 0
    pad_token_id = 0
    alpha0 = 0.5

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Architecture (Strictly synchronized with train_esolm.py) ---
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

def _get_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return torch.zeros(seq_len, seq_len, device=device).masked_fill(~mask, float('-inf'))

def _get_seq_mask(seq_len, device):
    L = seq_len
    q_idx  = torch.arange(2*L, device=device).unsqueeze(1).expand(2*L, 2*L)
    kv_idx = torch.arange(2*L, device=device).unsqueeze(0).expand(2*L, 2*L)
    
    # h_in = [x_shuffled, z_shuffled]
    x_flag_q  = (q_idx  >= L).long()
    x_flag_kv = (kv_idx >= L).long()
    
    q_idx2  = torch.where(x_flag_q  == 1, q_idx  - L, q_idx)
    kv_idx2 = torch.where(x_flag_kv == 1, kv_idx - L, kv_idx)

    # 1. Clean tokens see clean tokens: x_i attends x_<=i
    m_xx = (q_idx2 >= kv_idx2) & (x_flag_q == 0) & (x_flag_kv == 0)
    # 2. Teacher forcing: z_i attends x_<i (NO LEAKAGE)
    m_zx = (q_idx2 > kv_idx2)  & (x_flag_q == 1) & (x_flag_kv == 0)
    # 3. Noisy context: z_i attends z_<=i
    m_zz = (q_idx2 >= kv_idx2) & (x_flag_q == 1) & (x_flag_kv == 1)
    
    allowed = m_xx | m_zx | m_zz
    return torch.zeros(2*L, 2*L, device=device).masked_fill(~allowed, float('-inf'))

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        
    def forward(self, x, rotary, attn_mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Pull rotary from the shared embeddings passed into forward
        cos = rotary[..., :self.head_dim].unsqueeze(1)
        sin = rotary[..., self.head_dim:].unsqueeze(1)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None: att = att + attn_mask.unsqueeze(0).unsqueeze(0)
        att = F.softmax(att, dim=-1)
        out = self.dropout(att) @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = int(dim * 8 / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
        self.attn = Attention()
        self.ff = FeedForward(cfg.dim)
        
    def forward(self, x, rotary, attn_mask=None):
        x = x + self.attn(self.ln1(x), rotary, attn_mask)
        x = x + self.ff(self.ln2(x))
        return x

class EsoLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.dim)
        self.output_layer = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.output_layer.weight = self.token_emb.weight
        self.rotary_emb = RotaryEmbedding(cfg.dim // cfg.n_heads)

    def _sample_ordering(self, zt, shuffle_masks):
        masked = (zt == cfg.mask_token_id)          # [B, L]
        offsets = torch.rand(zt.shape, device=zt.device)
        if not shuffle_masks:
            B, L = zt.shape
            counts = masked.sum(dim=1, keepdim=True).float().clamp(min=1)
            cumsum = masked.float().cumsum(dim=1) - 1.0
            frac = cumsum / counts * 0.99
            offsets = torch.where(masked, frac, offsets)
        return (masked.float() + offsets).argsort(dim=1)

    def _sort(self, tensor, ordering):
        if tensor.dim() == 2: return torch.gather(tensor, dim=1, index=ordering)
        elif tensor.dim() == 3: return torch.gather(tensor, dim=1, index=ordering.unsqueeze(-1).expand_as(tensor))

    def forward(self, zt, x=None, diffusion_mode=True):
        seq_len = zt.shape[1]
        h_zt = self.token_emb(zt)
        cos_cached = self.rotary_emb.cos_cached[0, 0, :seq_len, :]
        sin_cached = self.rotary_emb.sin_cached[0, 0, :seq_len, :]
        rotary = torch.cat([cos_cached, sin_cached], dim=-1).unsqueeze(0).expand(zt.shape[0], -1, -1)
        
        if diffusion_mode:
            attn_mask = _get_causal_mask(seq_len, zt.device)
            ordering = self._sample_ordering(zt, shuffle_masks=True)
            h_in, rotary_in = self._sort(h_zt, ordering), self._sort(rotary, ordering)
        else:
            attn_mask = _get_seq_mask(seq_len, zt.device)
            ordering = self._sample_ordering(zt, shuffle_masks=False)
            h_x = self.token_emb(x)
            h_in = torch.cat([self._sort(h_x, ordering), self._sort(h_zt, ordering)], dim=1)
            rotary_shuffled = self._sort(rotary, ordering)
            rotary_in = torch.cat([rotary_shuffled, rotary_shuffled], dim=1)
            
        for block in self.blocks: 
            h_in = block(h_in, rotary_in, attn_mask=attn_mask)
            
        return self.output_layer(self.ln(h_in)), ordering

# --- Specialized Two-Phase Inference ---
@torch.no_grad()
def generate(model, sp, prompt, steps=16, temperature=1.0):
    model.eval()
    tokens = sp.encode(prompt)
    if len(tokens) > cfg.seq_len: tokens = tokens[-cfg.seq_len:]
    
    # 1. Prepare Initial masked sequence
    zt = torch.full((1, cfg.seq_len), cfg.mask_token_id, device=device)
    prompt_len = len(tokens)
    zt[0, :prompt_len] = torch.tensor(tokens, device=device)
    
    # --- PHASE 1: DIFFUSION (Parallel Sketching) ---
    for step in range(steps):
        is_masked = (zt == cfg.mask_token_id)
        if not is_masked.any(): break
        
        logits, ordering = model(zt, diffusion_mode=True)
        
        # Un-shuffle logits back to natural sequence order
        inv_ordering = torch.empty_like(ordering)
        inv_ordering.scatter_(1, ordering, torch.arange(cfg.seq_len, device=device).expand_as(ordering))
        logits_aligned = torch.gather(logits, 1, inv_ordering.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
        
        # Sampling with Temperature + Top-P
        probs = F.softmax(logits_aligned / temperature, dim=-1)
        
        # Top-P Filtering
        top_p = 0.95
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        sampled = torch.multinomial(probs[0], num_samples=1).squeeze(-1)
        
        # Linear annealing of masks
        threshold = cfg.alpha0 * (1 - (step / steps))
        to_unmask = torch.rand(1, cfg.seq_len, device=device) > threshold
        new_unmasked = to_unmask & is_masked
        zt[new_unmasked] = sampled.unsqueeze(0)[new_unmasked]

    # PHASE 2: Iterative Sequential Polishing (Teacher Forcing)
    # Multiple passes allow the model to correct its own grammar and naming slips.
    for polish_round in range(3):
        is_masked = (zt == cfg.mask_token_id)
        # In the first round, we polish everything. 
        # In subsequent rounds, we refine the whole sequence to ensure logic holds.
        logits, ordering = model(zt, x=zt, diffusion_mode=False) 
        logits_seq = logits[:, cfg.seq_len:, :]
        inv_ordering = torch.empty_like(ordering)
        inv_ordering.scatter_(1, ordering, torch.arange(cfg.seq_len, device=device).expand_as(ordering))
        logits_aligned = torch.gather(logits_seq, 1, inv_ordering.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
        
        # We update the sequence with the most likely tokens from the Teacher Forcing pass
        zt = torch.argmax(logits_aligned, dim=-1)
        
        # Keep the original prompt tokens frozen if they were provided
        if len(tokens) > 0:
            zt[0, :prompt_len] = torch.tensor(tokens[:prompt_len], device=device)

    return sp.decode(zt[0].tolist())

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor()
    sp.load('tinystories_bpe.model')
    
    # Dynamically set the mask token ID from the sp processor
    cfg.mask_token_id = sp.piece_to_id('[MASK]')
    
    model = EsoLM().to(device)
    
    import glob
    existing_dirs = glob.glob("checkpoints_[0-9][0-9][0-9][0-9]")
    if os.path.exists("esolm_checkpoint.pt"):
        checkpoint_path = "esolm_checkpoint.pt"
    elif existing_dirs:
        latest_dir = max(existing_dirs, key=lambda d: int(d.split('_')[-1]))
        checkpoint_path = os.path.join(latest_dir, "esolm_faithful.pt")
    else:
        checkpoint_path = "checkpoints_may10-828pm/esolm_faithful.pt"
    
    try:
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            print(f"Model loaded successfully from {checkpoint_path}")
        else:
            print("No checkpoint found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Architecture mismatch detected. Please delete old .pt files and re-train.")
        
    while True:
        prompt = input("\nEnter prompt (or 'q' to quit): ")
        if prompt.lower() == 'q': break
        output = generate(model, sp, prompt)
        print("-" * 30)
        print(output)
        print("-" * 30)
