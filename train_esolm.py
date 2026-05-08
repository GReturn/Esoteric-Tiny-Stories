import os
import math
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
@dataclass
class Config:
    vocab_size = 8000
    seq_len = 128
    n_layers = 6
    n_heads = 6
    dim = 384
    dropout = 0.1
    batch_size = 16
    lr = 3e-4
    max_steps = 25000
    mask_token_id = 0
    pad_token_id = 0

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Preparation ---
def prepare_dataset():
    if not os.path.exists("tinystories.txt"):
        print("Streaming TinyStories (Safer for Windows)...")
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        num_stories = 50000
        with open("tinystories.txt", "w", encoding="utf-8") as f:
            for i, item in enumerate(dataset):
                f.write(item['text'] + "\n\n")
                if (i + 1) >= num_stories: break
                if (i + 1) % 5000 == 0: print(f"Downloaded {i+1} stories...")
        print("Done! Saved to tinystories.txt")

    if not os.path.exists("tinystories_bpe.model"):
        print("Training SentencePiece tokenizer...")
        spm.SentencePieceTrainer.train(
            input='tinystories.txt', 
            model_prefix='tinystories_bpe', 
            vocab_size=cfg.vocab_size,
            user_defined_symbols=['[MASK]'],
            pad_id=3, unk_id=0, bos_id=1, eos_id=2
        )

    sp = spm.SentencePieceProcessor()
    sp.load('tinystories_bpe.model')
    
    with open("tinystories.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Simple character-level fallback if BPE encoding is too slow for whole file
    # But for 50k stories (~50MB), it should be fine.
    print("Encoding dataset...")
    data = torch.tensor(sp.encode(text), dtype=torch.long)
    return data, sp

# --- Model Components ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

def apply_rotary(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1).reshape_as(x)
    return x * cos + rotated * sin

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
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1,2)
        
        cos, sin = self.rope(L)
        cos, sin = cos[None,None,:,:], sin[None,None,:,:]
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        # Stable attention for Windows
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            att = att + attn_bias.to(q.dtype)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,L,D)
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
        for block in self.blocks:
            h = block(h, attn_bias=attn_bias)
        return self.head(self.ln(h))

# --- Training Logic ---
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
    x = torch.stack([data[i:i+cfg.seq_len] for i in ix])
    return x.to(device)

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

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Setup
    data, sp = prepare_dataset()
    model = EsoLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    print(f"Starting training on {device}...")
    pbar = tqdm(range(cfg.max_steps))
    
    for step in pbar:
        model.train()
        x_clean = get_batch(data)
        B, L = x_clean.shape
        
        # Eso-LM B Diffusion Loss
        t = torch.randint(1, L, (B,), device=device)
        is_masked = torch.bernoulli((t.float() / L).unsqueeze(1).expand(B, L)).bool()
        
        x_corrupt = x_clean.clone()
        x_corrupt[is_masked] = cfg.mask_token_id
        
        if not is_masked.any():
            loss = (model(x_corrupt) * 0).sum()
        else:
            attn_bias = get_esolm_b_diffusion_mask(is_masked)
            logits = model(x_corrupt, attn_bias=attn_bias)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), x_clean.view(-1), reduction='none')
            loss = loss[is_masked.view(-1)].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}")
            # Quick sample
            sample = generate_diffusion(model, sp, "Once upon a time,")
            print(f"\nStep {step} Sample: {sample}\n")
            model.train()

    # Final Save
    torch.save(model.state_dict(), "esolm_tinystories.pt")
    print("Training complete. Model saved to esolm_tinystories.pt")
