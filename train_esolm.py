import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- Configuration (Optimized for 8GB VRAM) ---
@dataclass
class Config:
    vocab_size = 8000
    seq_len = 128
    n_layers = 8
    n_heads = 8
    dim = 512
    dropout = 0.05
    batch_size = 16
    lr = 3e-4
    warmup_steps = 2000
    max_steps = 25000
    eval_interval = 500
    eval_iters = 20
    mask_token_id = 0
    pad_token_id = 0

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_lr(it):
    if it < cfg.warmup_steps:
        return cfg.lr * it / cfg.warmup_steps
    if it > cfg.max_steps:
        return cfg.lr * 0.1
    decay_ratio = (it - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.lr * (0.1 + 0.9 * coeff)

# --- Model Components (With KV Caching) ---
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin, position_offset=0):
    cos = cos[:, :, position_offset:position_offset+x.shape[2], :]
    sin = sin[:, :, position_offset:position_offset+x.shape[2], :]
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

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.head_dim)
        
    def forward(self, x, attn_bias=None, kv_cache=None, position_offset=0):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = apply_rotary(q, self.rope.cos_cached, self.rope.sin_cached, position_offset)
        k = apply_rotary(k, self.rope.cos_cached, self.rope.sin_cached, position_offset)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            
        new_kv_cache = (k, v)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None: 
            att = att + attn_bias.to(q.dtype)
            
        att = F.softmax(att, dim=-1)
        out = self.dropout(att) @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out), new_kv_cache

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
        self.attn = Attention()
        self.ff = nn.Sequential(
            nn.Linear(cfg.dim, 4 * cfg.dim), nn.GELU(),
            nn.Linear(4 * cfg.dim, cfg.dim), nn.Dropout(cfg.dropout)
        )
        
    def forward(self, x, attn_bias=None, kv_cache=None, position_offset=0):
        attn_out, new_kv_cache = self.attn(self.ln1(x), attn_bias, kv_cache, position_offset)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_kv_cache

class EsoLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size)
        
    def forward(self, x, attn_bias=None, kv_caches=None, position_offset=0):
        h = self.token_emb(x)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, new_cache_i = block(h, attn_bias=attn_bias, kv_cache=cache_i, position_offset=position_offset)
            new_caches.append(new_cache_i)
        return self.head(self.ln(h)), new_caches

# --- Unified Permutation Objective Logic ---
def sample_permutation_mask(B, L, device):
    """Samples permutation order, applies 30% curriculum, and identifies masked prefix."""
    # 30% natural order curriculum for stability
    if random.random() < 0.3:
        perm = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    else:
        perm = torch.argsort(torch.rand(B, L, device=device), dim=-1)

    t = torch.randint(1, L, (B,), device=device)
    rank = torch.argsort(perm, dim=-1)
    
    visible = rank < t.unsqueeze(1)
    is_masked = ~visible
    return perm, rank, visible, is_masked

def get_permutation_attention_bias(rank, visible):
    """
    Token i can attend to token j iff j is visible OR rank_j <= rank_i
    """
    B, L = rank.shape
    device = rank.device

    rank_i = rank.unsqueeze(2)
    rank_j = rank.unsqueeze(1)
    visible_j = visible.unsqueeze(1).expand(B, L, L)

    allowed = visible_j | (rank_j <= rank_i)
    
    # Always allow self-attention
    eye = torch.eye(L, device=device).bool()
    allowed = allowed | eye.unsqueeze(0)

    attn_bias = torch.zeros(B, L, L, device=device)
    attn_bias.masked_fill_(~allowed, float('-inf'))
    return attn_bias.unsqueeze(1)

def get_causal_mask(T, device):
    """Used strictly for prefixing during inference."""
    mask = torch.tril(torch.ones(T, T, device=device)).bool()
    attn_bias = torch.zeros(T, T, device=device).masked_fill(~mask, float('-inf'))
    return attn_bias.unsqueeze(0).unsqueeze(0)

def get_batch(data):
    ix = torch.randint(len(data) - cfg.seq_len, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.seq_len] for i in ix])
    return x.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x_clean = get_batch(data)
            B, L = x_clean.shape
            
            perm, rank, visible, is_masked = sample_permutation_mask(B, L, device)
            x_corrupt = x_clean.clone()
            x_corrupt[is_masked] = cfg.mask_token_id
            
            attn_bias = get_permutation_attention_bias(rank, visible)
            logits, _ = model(x_corrupt, attn_bias=attn_bias)
            
            loss_flat = F.cross_entropy(logits.view(-1, cfg.vocab_size), x_clean.view(-1), reduction='none')
            masked_loss = loss_flat[is_masked.view(-1)]
            
            # Prevent NaN if completely unmasked (e.g. t == L)
            losses[k] = masked_loss.mean().item() if masked_loss.numel() > 0 else 0.0
            
        out[split] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def generate_diffusion(model, sp, prompt, steps=32, temperature=0.0):
    """Optimized diffusion with exact KV Caching and progressive freezing."""
    model.eval()
    tokens = sp.encode(prompt)
    L_prompt = len(tokens)
    gen_len = cfg.seq_len - L_prompt
    
    prompt_tensor = torch.tensor([tokens], device=device)
    
    # PREFILL CACHE
    prompt_mask = get_causal_mask(L_prompt, device)
    _, kv_caches = model(prompt_tensor, attn_bias=prompt_mask)
    
    x_gen = torch.full((1, gen_len), cfg.mask_token_id, device=device)
    
    # DIFFUSION DECODING
    for step in range(steps):
        is_masked = (x_gen == cfg.mask_token_id)
        if not is_masked.any(): break
            
        # Unmasked tokens are visible. Masked are invisible.
        B, L_gen = x_gen.shape
        j_is_unmasked = (~is_masked).unsqueeze(1).expand(B, L_gen, L_gen)
        allowed = j_is_unmasked | torch.eye(L_gen, device=device).bool().unsqueeze(0)
        
        attn_bias_gen = torch.zeros(B, L_gen, L_gen, device=device).masked_fill_(~allowed, float('-inf')).unsqueeze(1)
        
        # Connect generation block back to cached prompt
        bias_prefix = torch.zeros(1, 1, L_gen, L_prompt, device=device)
        full_bias = torch.cat([bias_prefix, attn_bias_gen], dim=-1)
        
        logits, _ = model(x_gen, attn_bias=full_bias, kv_caches=kv_caches, position_offset=L_prompt)
        
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.distributions.Categorical(probs).sample()
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)
            sampled = torch.argmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
            
        # Propose sampling
        x_gen[is_masked] = sampled[is_masked]
        
        # Cosine scheduling (progressive freezing)
        remaining_ratio = 0.5 * (1 + math.cos(math.pi * step / steps))
        candidate_positions = torch.where(is_masked.view(-1))[0]
        candidate_conf = confidence.view(-1)[candidate_positions]
        
        num_to_keep_masked = int(remaining_ratio * candidate_conf.numel())
        if num_to_keep_masked > 0:
            keep_masked_idx = torch.topk(candidate_conf, num_to_keep_masked, largest=False).indices
            # Revert the lowest confidence tokens back to MASK
            x_gen.view(-1)[candidate_positions[keep_masked_idx]] = cfg.mask_token_id
            
    final_sequence = torch.cat([prompt_tensor, x_gen], dim=1)
    return sp.decode(final_sequence[0].tolist())

if __name__ == "__main__":
    if not os.path.exists("tinystories.txt"):
        print("Streaming TinyStories...")
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        with open("tinystories.txt", "w", encoding="utf-8") as f:
            for i, item in enumerate(dataset):
                f.write(item['text'] + "\n\n")
                if i >= 50000: break
                
    if not os.path.exists("tinystories_bpe.model"):
        spm.SentencePieceTrainer.train(input='tinystories.txt', model_prefix='tinystories_bpe', vocab_size=cfg.vocab_size, user_defined_symbols=['[MASK]'], pad_id=3, unk_id=0, bos_id=1, eos_id=2)
        
    sp = spm.SentencePieceProcessor()
    sp.load('tinystories_bpe.model')
    print("Encoding dataset...")
    import numpy as np
    with open("tinystories.txt", "r", encoding="utf-8") as f: data_text = f.read()
    tokens = sp.encode(data_text)
    all_data = torch.from_numpy(np.array(tokens, dtype=np.int32)).long()
    n = int(0.9 * len(all_data))
    train_data, val_data = all_data[:n], all_data[n:]

    model = EsoLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    
    train_losses, val_losses, steps_list = [], [], []
    
    print(f"Starting unified Eso-LM training on {device}...")
    scaler = torch.amp.GradScaler('cuda')
    pbar = tqdm(range(cfg.max_steps))
    
    try:
        for step in pbar:
            curr_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

            if step % cfg.eval_interval == 0:
                losses = estimate_loss(model, train_data, val_data)
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                steps_list.append(step)
                
                pbar.set_description(f"Step {step} | LR: {curr_lr:.2e} | Val Loss: {losses['val']:.4f}")
                print(f"\nSample: {generate_diffusion(model, sp, 'Once upon a time,')}\n")
                
                if step > 0:
                    torch.save(model.state_dict(), f"esolm_checkpoint.pt")

            # --- UNIFIED TRAINING STEP ---
            x_clean = get_batch(train_data)
            B, L = x_clean.shape
            
            # 1. Sample topology curriculum
            perm, rank, visible, is_masked = sample_permutation_mask(B, L, device)
            
            # 2. Corrupt inputs
            x_corrupt = x_clean.clone()
            x_corrupt[is_masked] = cfg.mask_token_id
            
            # 3. Factorized attention map
            attn_bias = get_permutation_attention_bias(rank, visible)
            
            with torch.amp.autocast('cuda'):
                logits, _ = model(x_corrupt, attn_bias=attn_bias)
                loss_flat = F.cross_entropy(logits.view(-1, cfg.vocab_size), x_clean.view(-1), reduction='none')
                
                # Compute loss exclusively on dynamically masked tokens
                masked_loss = loss_flat[is_masked.view(-1)]
                loss = masked_loss.mean() if masked_loss.numel() > 0 else loss_flat.mean() * 0.0

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")

    finally:
        torch.save(model.state_dict(), "esolm_tinystories.pt")
        
        if len(steps_list) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(steps_list, train_losses, label='Train NELBO Loss', color='blue', alpha=0.6)
            plt.plot(steps_list, val_losses, label='Val NELBO Loss', color='red', linewidth=2)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Eso-LM Unified Training Progress')
            plt.grid(True, alpha=0.3)
            plt.savefig('unified_loss_plot.png')
            print(f"Final model saved. Plot saved with {len(steps_list)} eval points.")