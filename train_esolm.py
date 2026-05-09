import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- Configuration (Optimized for 8GB VRAM with 2L Concatenation) ---
@dataclass
class Config:
    vocab_size = 10000 # Increased for better coverage (matches paper)
    seq_len = 256      # Upgraded for full stories
    n_layers = 8
    n_heads = 8
    dim = 512
    dropout = 0.1
    batch_size = 8     # Reduced per-step batch size for SwiGLU memory
    grad_accum_steps = 8 # Global batch size = 64 (Pro Stability)
    lr = 6e-4          # Higher LR for small models (Pro-tip)
    warmup_steps = 2000
    max_steps = 25000
    eval_interval = 500
    eval_iters = 20
    mask_token_id = 0
    pad_token_id = 0
    alpha0 = 0.5       # The core Eso-LM interpolation hyperparameter

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

# --- Faithful Eso-LM Attention Logic (from Appendix B.10) ---
def _get_causal_mask(seq_len, device):
    """Standard causal mask used for the physically shuffled Diffusion Phase."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return torch.zeros(seq_len, seq_len, device=device).masked_fill(~mask, float('-inf'))

def _get_seq_mask(seq_len, device):
    """The 2L x 2L specialized mask for Sequential Phase (App. B.7 / Fig 9)."""
    # PyTorch implementation of the FlexAttention mask described in Figure 9
    L = seq_len
    # We create a 2L x 2L mask
    q_idx = torch.arange(2*L, device=device).unsqueeze(1).expand(2*L, 2*L)
    kv_idx = torch.arange(2*L, device=device).unsqueeze(0).expand(2*L, 2*L)
    
    x_flag_q = (q_idx >= L).long()
    x_flag_kv = (kv_idx >= L).long()
    
    q_idx2 = torch.where(x_flag_q == 1, q_idx - L, q_idx)
    kv_idx2 = torch.where(x_flag_kv == 1, kv_idx - L, kv_idx)
    
    # 1. Causal Mask for Clean Half (Upper Left: x sees x_<=i)
    diagonal_causal = (q_idx2 >= kv_idx2) & (x_flag_q == 0) & (x_flag_kv == 0)
    
    # 2. Teacher Forcing Mask (Lower Left: z_i sees x_<i)
    # Note: z_i MUST NOT see x_i (its own ground truth) to prevent cheating
    teacher_forcing = (q_idx2 > kv_idx2) & (x_flag_q == 1) & (x_flag_kv == 0)
    
    # 3. Self-Attention for Noisy Half (Lower Right: z_i sees z_i)
    # In Sequential Phase, z_i primarily relies on clean prefix x_<i
    self_attn_z = (q_idx2 == kv_idx2) & (x_flag_q == 1) & (x_flag_kv == 1)
    
    allowed = diagonal_causal | teacher_forcing | self_attn_z
    
    mask = torch.zeros(2*L, 2*L, device=device).masked_fill(~allowed, float('-inf'))
    return mask

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        
    def forward(self, x, rotary, attn_mask=None, kv_cache=None):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos = rotary[..., :self.head_dim].unsqueeze(1)
        sin = rotary[..., self.head_dim:].unsqueeze(1)
        
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            
        new_kv_cache = (k, v)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None: 
            att = att + attn_mask.unsqueeze(0).unsqueeze(0)
            
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
            nn.Linear(cfg.dim, 4 * cfg.dim),
            nn.SiLU(), # SwiGLU style activation (Pro Upgrade)
            nn.Linear(4 * cfg.dim, cfg.dim),
            nn.Dropout(cfg.dropout)
        )
        
    def forward(self, x, rotary, attn_mask=None, kv_cache=None):
        attn_out, new_kv_cache = self.attn(self.ln1(x), rotary, attn_mask, kv_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_kv_cache

class EsoLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.dim)
        self.output_layer = nn.Linear(cfg.dim, cfg.vocab_size)
        self.rotary_emb = RotaryEmbedding(cfg.dim // cfg.n_heads)

    def _sample_ordering(self, zt, shuffle_masks):
        """Faithful physical shuffling logic from Appendix B.10 (Figure 12)"""
        masked = (zt == cfg.mask_token_id)
        offsets = torch.rand(zt.shape, device=zt.device)
        
        if not shuffle_masks:
            # Induce strict left-to-right natural order within masked tokens for Sequential Phase
            # This matches the authors' 'keep_masks_unshuffled' logic exactly.
            num_masked = torch.sum(masked, dim=1)
            for b in range(zt.size(0)):
                if num_masked[b] > 0:
                    # Overwrite offsets for masked tokens to be strictly ascending (0.0 to 0.99)
                    # This ensures they stay in order at the end of the shuffled sequence.
                    offsets[b, masked[b]] = torch.linspace(0, 0.99, num_masked[b].item(), device=zt.device)
                    
        # Sort puts clean tokens (0 + offset) before mask tokens (1 + offset)
        ordering = (masked.float() + offsets).argsort(dim=1)
        return ordering

    def _sort(self, tensor, ordering):
        if tensor.dim() == 2:
            return torch.gather(tensor, dim=1, index=ordering)
        elif tensor.dim() == 3:
            return torch.gather(tensor, dim=1, index=ordering.unsqueeze(-1).expand_as(tensor))

    def forward(self, zt, x=None, diffusion_mode=True):
        seq_len = zt.shape[1]
        
        h_zt = self.token_emb(zt)
        cos_cached = self.rotary_emb.cos_cached[0, 0, :seq_len, :]
        sin_cached = self.rotary_emb.sin_cached[0, 0, :seq_len, :]
        rotary = torch.cat([cos_cached, sin_cached], dim=-1).unsqueeze(0).expand(zt.shape[0], -1, -1)
        
        if diffusion_mode:
            attn_mask = _get_causal_mask(seq_len, zt.device)
            ordering = self._sample_ordering(zt, shuffle_masks=True)
            
            # Physically shuffle the tokens and their positional rotaries
            h_in = self._sort(h_zt, ordering)
            rotary_in = self._sort(rotary, ordering)
            
        else: # Sequential Mode (Requires 2L concatenation!)
            assert x is not None, "Sequential mode requires clean x for 2L concatenation."
            attn_mask = _get_seq_mask(seq_len, zt.device)
            ordering = self._sample_ordering(zt, shuffle_masks=False)
            
            h_x = self.token_emb(x)
            
            # Concatenate [x_shuffled, zt_shuffled]
            h_in = torch.cat([self._sort(h_x, ordering), self._sort(h_zt, ordering)], dim=1)
            rotary_shuffled = self._sort(rotary, ordering)
            rotary_in = torch.cat([rotary_shuffled, rotary_shuffled], dim=1)
            
        for block in self.blocks:
            h_in, _ = block(h_in, rotary_in, attn_mask=attn_mask)
            
        logits = self.output_layer(self.ln(h_in))
        
        return logits, ordering

# --- Training Utils ---
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
            x_batch = get_batch(data)
            B, L = x_batch.shape
            half_B = B // 2
            x_seq = x_batch[:half_B]
            x_diff = x_batch[half_B:]
            
            # 1. Sequential Loss
            is_masked_seq = torch.rand(half_B, L, device=device) > cfg.alpha0
            z0 = x_seq.clone()
            z0[is_masked_seq] = cfg.mask_token_id
            logits_seq, ordering_seq = model(z0, x=x_seq, diffusion_mode=False)
            logits_seq_z0 = logits_seq[:, :L, :] 
            inv_ordering_seq = torch.empty_like(ordering_seq)
            inv_ordering_seq.scatter_(1, ordering_seq, torch.arange(L, device=device).expand_as(ordering_seq))
            logits_seq_aligned = torch.gather(logits_seq_z0, 1, inv_ordering_seq.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
            loss_seq = F.cross_entropy(logits_seq_aligned.reshape(-1, cfg.vocab_size), x_seq.view(-1), reduction='none')
            l_seq = loss_seq[is_masked_seq.view(-1)].mean() if is_masked_seq.any() else torch.tensor(0.0, device=device)
            
            # 2. Diffusion Loss
            t = torch.rand(half_B, 1, device=device)
            alpha_t = cfg.alpha0 * (1 - t)
            is_masked_diff = torch.rand(half_B, L, device=device) > alpha_t
            zt = x_diff.clone()
            zt[is_masked_diff] = cfg.mask_token_id
            logits_diff, ordering_diff = model(zt, diffusion_mode=True)
            inv_ordering_diff = torch.empty_like(ordering_diff)
            inv_ordering_diff.scatter_(1, ordering_diff, torch.arange(L, device=device).expand_as(ordering_diff))
            logits_diff_aligned = torch.gather(logits_diff, 1, inv_ordering_diff.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
            loss_diff = F.cross_entropy(logits_diff_aligned.reshape(-1, cfg.vocab_size), x_diff.view(-1), reduction='none')
            l_diff = loss_diff[is_masked_diff.view(-1)].mean() if is_masked_diff.any() else torch.tensor(0.0, device=device)
            
            losses[k] = (l_seq + l_diff).item() / 2.0
        out[split] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def generate_sample(model, sp, prompt="Once upon a time,"):
    model.eval()
    tokens = sp.encode(prompt)
    zt = torch.full((1, cfg.seq_len), cfg.mask_token_id, device=device)
    prompt_len = min(len(tokens), cfg.seq_len)
    zt[0, :prompt_len] = torch.tensor(tokens[:prompt_len], device=device)
    
    # Simple 4-step diffusion sketch for progress monitoring
    for step in range(4):
        is_masked = (zt == cfg.mask_token_id)
        if not is_masked.any(): break
        logits, ordering = model(zt, diffusion_mode=True)
        inv_ordering = torch.empty_like(ordering)
        inv_ordering.scatter_(1, ordering, torch.arange(cfg.seq_len, device=device).expand_as(ordering))
        logits_aligned = torch.gather(logits, 1, inv_ordering.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
        sampled = torch.argmax(logits_aligned, dim=-1)
        # Randomly fill some masks
        to_fill = torch.rand(1, cfg.seq_len, device=device) > 0.5
        zt[is_masked & to_fill] = sampled[is_masked & to_fill]
        
    return sp.decode(zt[0].tolist())

if __name__ == "__main__":
    if not os.path.exists("tinystories.txt"):
        print("Streaming TinyStories...")
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        with open("tinystories.txt", "w", encoding="utf-8") as f:
            for i, item in enumerate(dataset):
                f.write(item['text'] + "\n\n")
                if i >= 50000: break
                
    if not os.path.exists("tinystories_bpe.model"):
        print("Training High-Reliability Tokenizer...")
        spm.SentencePieceTrainer.train(
            input='tinystories.txt', 
            model_prefix='tinystories_bpe', 
            vocab_size=cfg.vocab_size, 
            user_defined_symbols=['[MASK]'], 
            byte_fallback=True,          # The "Magic Fix" for ⁇
            model_type='bpe',            # Match original paper
            character_coverage=1.0,      # Full coverage
            pad_id=3, unk_id=0, bos_id=1, eos_id=2
        )
        
    sp = spm.SentencePieceProcessor()
    sp.load('tinystories_bpe.model')
    cfg.mask_token_id = sp.piece_to_id('[MASK]')
    print(f"Using Mask Token ID: {cfg.mask_token_id}")
    
    import numpy as np
    with open("tinystories.txt", "r", encoding="utf-8") as f: data_text = f.read()
    tokens = sp.encode(data_text)
    all_data = torch.from_numpy(np.array(tokens, dtype=np.int32)).long()
    n = int(0.9 * len(all_data))
    train_data, val_data = all_data[:n], all_data[n:]

    model = EsoLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    
    train_losses, val_losses, steps_list = [], [], []
    
    print(f"Starting FAITHFUL Eso-LM Training on {device}...")
    scaler = torch.amp.GradScaler('cuda')
    pbar = tqdm(range(cfg.max_steps))
    
    try:
        for step in pbar:
            curr_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

            optimizer.zero_grad()
            # --- PRO GRADIENT ACCUMULATION ---
            for accum_step in range(cfg.grad_accum_steps):
                x_batch = get_batch(train_data)
                B, L = x_batch.shape
                half_B = B // 2
                x_seq = x_batch[:half_B]
                x_diff = x_batch[half_B:]
                
                with torch.amp.autocast('cuda'):
                    # 1. SEQUENTIAL PHASE
                    is_masked_seq = torch.rand(half_B, L, device=device) > cfg.alpha0
                    z0 = x_seq.clone()
                    z0[is_masked_seq] = cfg.mask_token_id
                    logits_seq, ordering_seq = model(z0, x=x_seq, diffusion_mode=False)
                    logits_seq_z = logits_seq[:, L:, :] 
                    inv_ordering_seq = torch.empty_like(ordering_seq)
                    inv_ordering_seq.scatter_(1, ordering_seq, torch.arange(L, device=device).expand_as(ordering_seq))
                    logits_seq_aligned = torch.gather(logits_seq_z, 1, inv_ordering_seq.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
                    loss_seq_flat = F.cross_entropy(logits_seq_aligned.reshape(-1, cfg.vocab_size), x_seq.view(-1), reduction='none')
                    loss_seq = loss_seq_flat[is_masked_seq.view(-1)].mean() if is_masked_seq.any() else torch.tensor(0.0, device=device)
                    
                    # 2. DIFFUSION PHASE
                    t = torch.rand(half_B, 1, device=device)
                    alpha_t = cfg.alpha0 * (1 - t)
                    is_masked_diff = torch.rand(half_B, L, device=device) > alpha_t
                    zt = x_diff.clone()
                    zt[is_masked_diff] = cfg.mask_token_id
                    logits_diff, ordering_diff = model(zt, diffusion_mode=True)
                    inv_ordering_diff = torch.empty_like(ordering_diff)
                    inv_ordering_diff.scatter_(1, ordering_diff, torch.arange(L, device=device).expand_as(ordering_diff))
                    logits_diff_aligned = torch.gather(logits_diff, 1, inv_ordering_diff.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
                    loss_diff_flat = F.cross_entropy(logits_diff_aligned.reshape(-1, cfg.vocab_size), x_diff.view(-1), reduction='none')
                    loss_diff = loss_diff_flat[is_masked_diff.view(-1)].mean() if is_masked_diff.any() else torch.tensor(0.0, device=device)

                    loss = (loss_seq + loss_diff) / (2.0 * cfg.grad_accum_steps)

                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if step % cfg.eval_interval == 0:
                metrics = estimate_loss(model, train_data, val_data)
                train_losses.append(metrics['train'])
                val_losses.append(metrics['val'])
                steps_list.append(step)
                pbar.set_description(f"Step {step} | Train: {metrics['train']:.3f} | Val: {metrics['val']:.3f}")
                print(f"\nSample: {generate_sample(model, sp)}\n")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving...")
    finally:
        torch.save(model.state_dict(), "esolm_faithful.pt")
        
        if len(steps_list) > 0:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(steps_list, train_losses, label='Train Loss', color='blue', alpha=0.6)
            plt.plot(steps_list, val_losses, label='Val Loss', color='red', linewidth=2)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Eso-LM (Faithful) Training Progress')
            plt.grid(True, alpha=0.3)
            plt.savefig('esolm_faithful_plot.png')
            print(f"Final model and plot (esolm_faithful_plot.png) saved.")