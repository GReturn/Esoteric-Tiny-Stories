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

# --- Configuration ---
@dataclass
class Config:
    vocab_size = 8000
    seq_len = 128
    n_layers = 6
    n_heads = 6
    dim = 384
    dropout = 0.1
    batch_size = 256
    lr = 3e-4
    max_steps = 2000 # Adjusted for batch size 256
    eval_interval = 200
    eval_iters = 20 # Much faster validation
    ar_weight = 1.0
    mdm_weight = 1.0
    alpha0 = 0.5
    mask_token_id = 0
    pad_token_id = 0

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Components ---
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
        out = self.dropout(att) @ v
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

# --- Utils ---
def get_causal_mask(T, device):
    """Standard causal mask for Autoregressive training."""
    mask = torch.tril(torch.ones(T, T, device=device)).bool()
    attn_bias = torch.zeros(T, T, device=device).masked_fill(~mask, float('-inf'))
    return attn_bias.unsqueeze(0).unsqueeze(0) # (1, 1, T, T)

def get_esolm_b_diffusion_mask(is_masked):
    B, T = is_masked.shape
    device = is_masked.device
    rand_keys = torch.rand(B, T, device=device)
    sort_keys = rand_keys + is_masked.float()
    sigma_inv = torch.argsort(torch.argsort(sort_keys, dim=1), dim=1)
    allowed = sigma_inv.unsqueeze(2) >= sigma_inv.unsqueeze(1)
    attn_bias = torch.zeros(B, T, T, device=device).masked_fill_(~allowed, float('-inf'))
    return attn_bias.unsqueeze(1)

def get_batch(data):
    ix = torch.randint(len(data) - cfg.seq_len, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.seq_len] for i in ix])
    return x.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses_ar = torch.zeros(cfg.eval_iters)
        losses_mdm = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x_clean = get_batch(data)
            B, L = x_clean.shape
            
            # AR Loss
            causal_bias = get_causal_mask(L, device)
            logits_ar = model(x_clean, attn_bias=causal_bias)
            loss_ar = F.cross_entropy(logits_ar[:, :-1, :].reshape(-1, cfg.vocab_size), x_clean[:, 1:].reshape(-1))
            losses_ar[k] = loss_ar.item()
            
            # MDM Loss
            t = torch.randint(1, L, (B,), device=device)
            is_masked = torch.bernoulli((t.float() / L).unsqueeze(1).expand(B, L)).bool()
            x_corrupt = x_clean.clone()
            x_corrupt[is_masked] = cfg.mask_token_id
            if is_masked.any():
                attn_bias = get_esolm_b_diffusion_mask(is_masked)
                logits_mdm = model(x_corrupt, attn_bias=attn_bias)
                loss_mdm = F.cross_entropy(logits_mdm.view(-1, cfg.vocab_size), x_clean.view(-1), reduction='none')
                losses_mdm[k] = loss_mdm[is_masked.view(-1)].mean().item()
            else:
                losses_mdm[k] = 0
                
        out[split] = (cfg.ar_weight * losses_ar.mean() + cfg.mdm_weight * losses_mdm.mean())
        out[f"{split}_ar"] = losses_ar.mean()
        out[f"{split}_mdm"] = losses_mdm.mean()
    model.train()
    return out

@torch.no_grad()
def generate_diffusion(model, sp, prompt, steps=32, temperature=0.0):
    model.eval()
    tokens = sp.encode(prompt)
    x = torch.tensor([tokens], device=device)
    B, L0 = x.shape
    gen_len = cfg.seq_len - L0
    x = torch.cat([x, torch.full((1, gen_len), cfg.mask_token_id, device=device)], dim=1)
    for step in range(steps):
        is_masked = (x == cfg.mask_token_id)
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
        remaining_ratio = 1 - (step + 1) / steps
        k = int(remaining_ratio * gen_len)
        if k > 0:
            masked_conf = confidence.clone()
            masked_conf[~is_masked] = 999
            idx = torch.topk(masked_conf.view(-1), k, largest=False).indices
            x.view(-1)[idx] = cfg.mask_token_id
    return sp.decode(x[0].tolist())

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
    with open("tinystories.txt", "r", encoding="utf-8") as f: data_text = f.read()
    all_data = torch.tensor(sp.encode(data_text), dtype=torch.long)
    n = int(0.9 * len(all_data))
    train_data, val_data = all_data[:n], all_data[n:]

    model = EsoLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    train_losses, val_losses = [], []
    val_ar_losses, val_mdm_losses = [], []
    steps_list = []
    
    print(f"Starting hybrid training (AR + MDM) on {device}...")
    print("Graphs will be saved to loss_plot.png (Safe to interrupt with Ctrl+C)")
    
    pbar = tqdm(range(cfg.max_steps))
    try:
        for step in pbar:
            if step % cfg.eval_interval == 0:
                losses = estimate_loss(model, train_data, val_data)
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                val_ar_losses.append(losses['val_ar'])
                val_mdm_losses.append(losses['val_mdm'])
                steps_list.append(step)
                
                pbar.set_description(f"Step {step} | AR: {losses['val_ar']:.4f} | MDM: {losses['val_mdm']:.4f}")
                print(f"\nSample: {generate_diffusion(model, sp, 'Once upon a time,')}\n")

            # Hybrid Step: Split batch into AR and MDM
            x_all = get_batch(train_data)
            B, L = x_all.shape
            half_B = B // 2
            
            # 1. AR component
            x_ar = x_all[:half_B]
            causal_bias = get_causal_mask(L, device)
            logits_ar = model(x_ar, attn_bias=causal_bias)
            loss_ar = F.cross_entropy(logits_ar[:, :-1, :].reshape(-1, cfg.vocab_size), x_ar[:, 1:].reshape(-1))
            
            # 2. MDM component
            x_mdm_clean = x_all[half_B:]
            t = torch.randint(1, L, (half_B,), device=device)
            is_masked = torch.bernoulli((t.float() / L).unsqueeze(1).expand(half_B, L)).bool()
            x_corrupt = x_mdm_clean.clone()
            x_corrupt[is_masked] = cfg.mask_token_id
            
            if is_masked.any():
                attn_bias = get_esolm_b_diffusion_mask(is_masked)
                logits_mdm = model(x_corrupt, attn_bias=attn_bias)
                loss_mdm_flat = F.cross_entropy(logits_mdm.view(-1, cfg.vocab_size), x_mdm_clean.view(-1), reduction='none')
                loss_mdm = loss_mdm_flat[is_masked.view(-1)].mean()
            else:
                loss_mdm = 0.0 * loss_ar
                
            loss = cfg.ar_weight * loss_ar + cfg.mdm_weight * loss_mdm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")

    finally:
        torch.save(model.state_dict(), "esolm_tinystories.pt")
        
        if len(steps_list) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(steps_list, val_ar_losses, label='Val AR (Grammar)', color='blue')
            plt.plot(steps_list, val_mdm_losses, label='Val MDM (Denoising)', color='green')
            plt.plot(steps_list, val_losses, label='Total Val Loss', color='red', linestyle='--')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Eso-LM Faithful Training Progress')
            plt.grid(True, alpha=0.3)
            plt.savefig('loss_plot.png')
            print("Final model saved to esolm_tinystories.pt")
            print("Final loss plot saved to loss_plot.png")
