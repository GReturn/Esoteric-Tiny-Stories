import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import csv

# --- Configuration (Optimized for 8GB VRAM) ---
@dataclass
class Config:
    vocab_size = 10000 
    seq_len = 256      
    n_layers = 8
    n_heads = 8
    dim = 512
    dropout = 0.1
    batch_size = 32    
    grad_accum_steps = 2
    lr = 6e-4          
    warmup_steps = 2000
    max_steps = 5000
    eval_interval = 100
    eval_iters = 50
    mask_token_id = 0
    alpha0 = 0.5       

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Rotary Embeddings ---
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

# --- Attention Masks ---
def _get_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return torch.zeros(seq_len, seq_len, device=device).masked_fill(~mask, float('-inf'))

def _get_seq_mask(seq_len, device):
    L = seq_len
    q_idx  = torch.arange(2*L, device=device).unsqueeze(1).expand(2*L, 2*L)
    kv_idx = torch.arange(2*L, device=device).unsqueeze(0).expand(2*L, 2*L)
    x_flag_q  = (q_idx  >= L).long()
    x_flag_kv = (kv_idx >= L).long()
    q_idx2  = torch.where(x_flag_q  == 1, q_idx  - L, q_idx)
    kv_idx2 = torch.where(x_flag_kv == 1, kv_idx - L, kv_idx)

    # Upper-left: z₀ self-causal (clean tokens, causal by σ)
    ul = (q_idx2 >= kv_idx2) & (x_flag_q == 0) & (x_flag_kv == 0)
    # Upper-right: z₀ sees x (teacher signal) — z_i attends to x_<i only
    ur = (q_idx2 > kv_idx2)  & (x_flag_q == 0) & (x_flag_kv == 1)
    # Lower-left: x sees z₀ clean tokens
    ll = (x_flag_q == 1) & (x_flag_kv == 0)
    # Lower-right: x sees x causally (for AR loss on x)
    lr = (q_idx2 >= kv_idx2) & (x_flag_q == 1) & (x_flag_kv == 1)

    allowed = ul | ur | ll | lr
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
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos = rotary[..., :self.head_dim].unsqueeze(1)
        sin = rotary[..., self.head_dim:].unsqueeze(1)
        
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            att = att + attn_mask
            
        att = F.softmax(att, dim=-1)
        out = self.dropout(att) @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = int(dim * 8 / 3)   # SwiGLU uses ~2.67× expansion
        hidden = (hidden + 63) // 64 * 64  # round to multiple of 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
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
        self.output_layer.weight = self.token_emb.weight   # tie weights
        self.rotary_emb = RotaryEmbedding(cfg.dim // cfg.n_heads)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _sample_ordering(self, zt, shuffle_masks):
        masked = (zt == cfg.mask_token_id)          # [B, L]
        offsets = torch.rand(zt.shape, device=zt.device)
        if not shuffle_masks:
            # Assign left-to-right fractional offsets to masked positions per row
            B, L = zt.shape
            counts = masked.sum(dim=1, keepdim=True).float().clamp(min=1)  # [B, 1]
            # rank of each masked token within its row
            cumsum = masked.float().cumsum(dim=1) - 1.0          # 0-indexed rank
            frac = cumsum / counts * 0.99                         # in [0, 0.99)
            offsets = torch.where(masked, frac, offsets)
        return (masked.float() + offsets).argsort(dim=1)

    def _sort(self, tensor, ordering):
        if tensor.dim() == 2:
            return torch.gather(tensor, dim=1, index=ordering)
        return torch.gather(tensor, dim=1, index=ordering.unsqueeze(-1).expand_as(tensor))

    def forward(self, zt, x=None, diffusion_mode=True):
        seq_len = zt.shape[1]
        h_zt = self.token_emb(zt)
        
        cos = self.rotary_emb.cos_cached[0, 0, :seq_len, :]
        sin = self.rotary_emb.sin_cached[0, 0, :seq_len, :]
        rotary = torch.cat([cos, sin], dim=-1).unsqueeze(0).expand(zt.shape[0], -1, -1)
        
        if diffusion_mode:
            attn_mask = _get_causal_mask(seq_len, zt.device)
            ordering = self._sample_ordering(zt, shuffle_masks=True)
            h_in = self._sort(h_zt, ordering)
            rotary_in = self._sort(rotary, ordering)
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

def sample_t_low_discrepancy(batch_size, device):
    u = torch.rand(1, device=device)
    offsets = torch.arange(batch_size, device=device).float() / batch_size
    return (u + offsets) % 1.0   # shape [batch_size]

# --- Zero-Latency Data Engine ---
def get_batch(data_tensor):
    ix = torch.randint(0, len(data_tensor) - cfg.seq_len, (cfg.batch_size,), device=device)
    offsets = torch.arange(cfg.seq_len, device=device)
    indices = ix.unsqueeze(1) + offsets
    return data_tensor[indices]

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(cfg.eval_iters):
            x_batch = get_batch(data)
            half_B = cfg.batch_size // 2
            x_seq, x_diff = x_batch[:half_B], x_batch[half_B:]
            
            with torch.amp.autocast('cuda'):
                # Sequential Loss
                is_m_seq = torch.rand(half_B, cfg.seq_len, device=device) > cfg.alpha0
                z0 = x_seq.clone()
                z0[is_m_seq] = cfg.mask_token_id
                logits_seq, ord_seq = model(z0, x=x_seq, diffusion_mode=False)
                
                inv_ord = torch.empty_like(ord_seq).scatter_(1, ord_seq, torch.arange(cfg.seq_len, device=device).expand_as(ord_seq))
                logits_seq_aligned = torch.gather(logits_seq[:, cfg.seq_len:, :], 1, inv_ord.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
                loss_seq = F.cross_entropy(logits_seq_aligned.reshape(-1, cfg.vocab_size), x_seq.view(-1), reduction='none')
                l_seq = loss_seq[is_m_seq.view(-1)].mean() if is_m_seq.any() else torch.tensor(0.0, device=device)
                
                # Diffusion Loss
                t = sample_t_low_discrepancy(half_B, device).unsqueeze(1)
                alpha_t = cfg.alpha0 * (1 - t)
                is_m_diff = torch.rand(half_B, cfg.seq_len, device=device) > alpha_t
                zt = x_diff.clone()
                zt[is_m_diff] = cfg.mask_token_id
                logits_diff, ord_diff = model(zt, diffusion_mode=True)
                
                inv_ord_d = torch.empty_like(ord_diff).scatter_(1, ord_diff, torch.arange(cfg.seq_len, device=device).expand_as(ord_diff))
                logits_diff_aligned = torch.gather(logits_diff, 1, inv_ord_d.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
                
                # Apply NELBO weight alpha_prime / (1 - alpha_t)
                alpha_prime = cfg.alpha0
                weight = alpha_prime / (1.0 - alpha_t)
                
                loss_diff_raw = F.cross_entropy(logits_diff_aligned.reshape(-1, cfg.vocab_size), x_diff.view(-1), reduction='none').view(half_B, cfg.seq_len)
                l_diff = (loss_diff_raw * is_m_diff * weight).sum() / is_m_diff.sum().clamp(min=1)
                
                losses.append((l_seq + l_diff) / 2.0)
                
        out[split] = torch.stack(losses).mean().item()
    model.train()
    return out

@torch.no_grad()
def generate_sample(model, sp, prompt="Once upon a time"):
    model.eval()
    tokens = sp.encode(prompt) if prompt else []
    if len(tokens) > cfg.seq_len: tokens = tokens[-cfg.seq_len:]
    
    zt = torch.full((1, cfg.seq_len), cfg.mask_token_id, device=device)
    prompt_len = len(tokens)
    if prompt_len > 0:
        zt[0, :prompt_len] = torch.tensor(tokens, device=device)

    # Simple 4-step monitoring sketch
    for _ in range(4):
        is_m = (zt == cfg.mask_token_id)
        if not is_m.any(): break
        logits, ord = model(zt, diffusion_mode=True)
        inv_ord = torch.empty_like(ord).scatter_(1, ord, torch.arange(cfg.seq_len, device=device).expand_as(ord))
        logits_aligned = torch.gather(logits, 1, inv_ord.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
        sampled = torch.argmax(logits_aligned, dim=-1)
        mask_prob = torch.rand(1, cfg.seq_len, device=device) > 0.5
        zt[is_m & mask_prob] = sampled[is_m & mask_prob]
    model.train()
    return sp.decode(zt[0].tolist())

if __name__ == "__main__":
    # 1. Dataset Streaming
    if not os.path.exists("tinystories.txt"):
        print("Streaming TinyStories...")
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        with open("tinystories.txt", "w", encoding="utf-8") as f:
            for i, item in enumerate(dataset):
                f.write(item['text'] + "\n\n")
                if i >= 50000: break
                
    # 2. Tokenizer Training
    if not os.path.exists("tinystories_bpe.model"):
        print("Training Tokenizer...")
        spm.SentencePieceTrainer.train(
            input='tinystories.txt', 
            model_prefix='tinystories_bpe', 
            vocab_size=cfg.vocab_size, 
            user_defined_symbols=['[MASK]'], 
            byte_fallback=True, 
            model_type='bpe', 
            character_coverage=1.0, 
            pad_id=3, unk_id=0, bos_id=1, eos_id=2
        )
        
    sp = spm.SentencePieceProcessor()
    sp.load('tinystories_bpe.model')
    cfg.mask_token_id = sp.piece_to_id('[MASK]')
    
    # 3. Data Pre-loading (to GPU)
    with open("tinystories.txt", "r", encoding="utf-8") as f: 
        data_text = f.read()
    tokens = torch.from_numpy(np.array(sp.encode(data_text), dtype=np.int32)).long().to(device)
    n = int(0.9 * len(tokens))
    train_data, val_data = tokens[:n], tokens[n:]
    
    # 4. Model Initialization
    model = EsoLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    # 5. Training Loop
    print(f"Launching Zero-Latency GPU Engine on {device}...")
    train_losses, val_losses, steps_list = [], [], []
    pbar = tqdm(range(cfg.max_steps))
    
    try:
        for step in pbar:
            # Learning Rate Schedule
            if step < cfg.warmup_steps:
                lr = cfg.lr * step / cfg.warmup_steps
            else:
                progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
                lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
            
            for pg in optimizer.param_groups: 
                pg['lr'] = lr
                
            optimizer.zero_grad()
            
            for _ in range(cfg.grad_accum_steps):
                x_batch = get_batch(train_data)
                half_B = cfg.batch_size // 2
                x_seq, x_diff = x_batch[:half_B], x_batch[half_B:]
                
                with torch.amp.autocast('cuda'):
                    # Sequential Step
                    is_m_seq = torch.rand(half_B, cfg.seq_len, device=device) > cfg.alpha0
                    z0 = x_seq.clone()
                    z0[is_m_seq] = cfg.mask_token_id
                    logits_seq, ord_seq = model(z0, x=x_seq, diffusion_mode=False)
                    
                    inv_ord = torch.empty_like(ord_seq).scatter_(1, ord_seq, torch.arange(cfg.seq_len, device=device).expand_as(ord_seq))
                    logits_seq_aligned = torch.gather(logits_seq[:, cfg.seq_len:, :], 1, inv_ord.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
                    l_seq = F.cross_entropy(logits_seq_aligned.reshape(-1, cfg.vocab_size), x_seq.view(-1), reduction='none')[is_m_seq.view(-1)].mean() if is_m_seq.any() else torch.tensor(0.0, device=device)
                    
                    # Diffusion Step
                    t = sample_t_low_discrepancy(half_B, device).unsqueeze(1)
                    alpha_t = cfg.alpha0 * (1 - t)
                    is_m_diff = torch.rand(half_B, cfg.seq_len, device=device) > alpha_t
                    zt = x_diff.clone()
                    zt[is_m_diff] = cfg.mask_token_id
                    logits_diff, ord_diff = model(zt, diffusion_mode=True)
                    
                    inv_ord_d = torch.empty_like(ord_diff).scatter_(1, ord_diff, torch.arange(cfg.seq_len, device=device).expand_as(ord_diff))
                    logits_diff_aligned = torch.gather(logits_diff, 1, inv_ord_d.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
                    
                    # Apply NELBO weight alpha_prime / (1 - alpha_t)
                    alpha_prime = cfg.alpha0
                    weight = alpha_prime / (1.0 - alpha_t)
                    
                    loss_diff_raw = F.cross_entropy(logits_diff_aligned.reshape(-1, cfg.vocab_size), x_diff.view(-1), reduction='none').view(half_B, cfg.seq_len)
                    l_diff = (loss_diff_raw * is_m_diff * weight).sum() / is_m_diff.sum().clamp(min=1)
                    
                    loss = (l_seq + l_diff) / (2.0 * cfg.grad_accum_steps)
                    
                scaler.scale(loss).backward()
                
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Monitoring
            if step % cfg.eval_interval == 0:
                metrics = estimate_loss(model, train_data, val_data)
                train_losses.append(metrics['train'])
                val_losses.append(metrics['val'])
                steps_list.append(step)
                pbar.set_description(f"Step {step} | Loss: {metrics['val']:.3f} | gnorm: {grad_norm:.2f}")
                print(f"\nSample: {generate_sample(model, sp)}\n")
                
                with open('loss_log.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'train_loss', 'val_loss'])
                    for s, t_l, v_l in zip(steps_list, train_losses, val_losses):
                        writer.writerow([s, t_l, v_l])
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    finally:
        torch.save(model.state_dict(), "esolm_faithful.pt")
        if len(steps_list) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(steps_list, train_losses, label='Train')
            plt.plot(steps_list, val_losses, label='Val')
            plt.title("Eso-LM Faithful Training Progress")
            plt.legend()
            plt.savefig('esolm_faithful_plot.png')
            print("Final model and plot saved.")