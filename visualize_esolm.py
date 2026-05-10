import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
from train_esolm import EsoLM, Config, device
import re

cfg = Config()

def visualize_generation(model, sp, prompt="Once upon a time"):
    model.eval()
    tokens = sp.encode(prompt)
    zt = torch.full((1, cfg.seq_len), cfg.mask_token_id, device=device)
    prompt_len = min(len(tokens), cfg.seq_len)
    if prompt_len > 0:
        zt[0, :prompt_len] = torch.tensor(tokens[:prompt_len], device=device)

    print("\n" + "="*50)
    print(f"VISUALIZING ESO-LM GENERATION")
    print(f"Prompt: '{prompt}'")
    print("="*50 + "\n")

    # --- 1. THE START (All Masked) ---
    print(f"[STAGE 1: INITIAL STATE (100% Masked)]")
    print("-" * 30)
    print(" ".join(["[MASK]"] * 10) + " ... [256 tokens total]")
    print("-" * 30 + "\n")

    # --- 2. DIFFUSION (The Sketch) ---
    steps = 48
    for step in [12, 24, 48]:
        # We simulate the diffusion up to this step
        # Note: In a real run we do this step-by-step
        with torch.no_grad():
            for s in range(step):
                is_masked = (zt == cfg.mask_token_id)
                if not is_masked.any(): break
                logits, ordering = model(zt, diffusion_mode=True)
                
                inv_ordering = torch.empty_like(ordering)
                inv_ordering.scatter_(1, ordering, torch.arange(cfg.seq_len, device=device).expand_as(ordering))
                logits_aligned = torch.gather(logits, 1, inv_ordering.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
                
                probs = F.softmax(logits_aligned / 0.8, dim=-1)
                sampled = torch.multinomial(probs[0], num_samples=1).squeeze(-1)
                
                current_alpha = cfg.alpha0 * (1.0 - (s / steps)**2) 
                to_unmask = torch.rand(1, cfg.seq_len, device=device) > current_alpha
                new_unmasked = to_unmask & is_masked
                zt[new_unmasked] = sampled.unsqueeze(0)[new_unmasked]

        print(f"[STAGE 2: DIFFUSION PROGRESS {int(step/steps*100)}% (Global Sketching)]")
        print("-" * 30)
        raw = sp.decode(zt[0].tolist())
        print(re.sub(r'\s+', ' ', raw[:300]) + "...")
        print("-" * 30 + "\n")

    # --- 3. SEQUENTIAL POLISHING (Grammar Fix) ---
    print(f"[STAGE 3: SEQUENTIAL POLISHING (AR Refinement)]")
    for i in range(2):
        with torch.no_grad():
            logits, ordering = model(zt, x=zt, diffusion_mode=False) 
            logits_seq = logits[:, cfg.seq_len:, :]
            inv_ordering = torch.empty_like(ordering)
            inv_ordering.scatter_(1, ordering, torch.arange(cfg.seq_len, device=device).expand_as(ordering))
            logits_aligned = torch.gather(logits_seq, 1, inv_ordering.unsqueeze(-1).expand(-1, -1, cfg.vocab_size))
            
            # Use argmax to pick the best words during polishing
            sampled = torch.argmax(logits_aligned, dim=-1)
            zt = sampled # Sampled is already [1, 256]
            if prompt_len > 0:
                zt[0, :prompt_len] = torch.tensor(tokens[:prompt_len], device=device)

        print(f"  Refinement Round {i+1}:")
        polished_text = sp.decode(zt[0].tolist())
        print("  " + re.sub(r'\s+', ' ', polished_text[:300]) + "...")
        print("-" * 30)
    
    print("\n" + "="*50)
    print("FINAL COHERENT STORY:")
    print("="*50)
    final = sp.decode(zt[0].tolist())
    # Truncate at EOS
    if sp.piece_to_id('</s>') in zt[0]:
        final = sp.decode(zt[0, :zt[0].tolist().index(sp.piece_to_id('</s>'))].tolist())
    print(re.sub(r'\s+', ' ', final))
    print("="*50 + "\n")

if __name__ == "__main__":
    ckpt_path = "checkpoints_0003/esolm_faithful.pt"
    if not os.path.exists(ckpt_path):
        print(f"Model not found at {ckpt_path}")
    else:
        sp = spm.SentencePieceProcessor()
        sp.load('tinystories_bpe.model')
        model = EsoLM().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        visualize_generation(model, sp)
