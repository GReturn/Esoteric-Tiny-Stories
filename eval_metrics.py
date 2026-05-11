import torch
import torch.nn.functional as F
import time
import os
import sentencepiece as spm
from train_esolm import EsoLM, Config, device, get_batch

# 1. Configuration & Constants
cfg = Config()
checkpoint_path = r'd:\_src\ML\is-gpt\checkpoints_0003\esolm_faithful.pt'
model_path = 'tinystories_bpe.model'

# 2. Setup Tokenizer
sp = spm.SentencePieceProcessor()
sp.load(model_path)
cfg.mask_token_id = sp.piece_to_id('[MASK]')

# 3. Load Data for Evaluation
# Note: We need the actual data to run evaluation. 
# We replicate the loading logic from train_esolm.py
print("Loading data for evaluation...")
with open("tinystories.txt", "r", encoding="utf-8") as f:
    stories = [line.strip() for line in f if line.strip()]

all_tokens = []
eos_id = 2 
for story in stories[:1000]: # Just take a subset for quick eval
    all_tokens.extend(sp.encode(story))
    all_tokens.append(eos_id)

eval_tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)

# 4. Load Model
model = EsoLM().to(device)
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print("Error: No checkpoint found!")
    exit()

model.eval()

@torch.no_grad()
def run_evaluation(num_batches=20):
    top1_accs = []
    mask_recon_rates = []
    
    print(f"Running evaluation on {num_batches} batches...")
    
    for _ in range(num_batches):
        x_batch = get_batch(eval_tokens)
        half_B = cfg.batch_size // 2
        x_seq, x_diff = x_batch[:half_B], x_batch[half_B:]
        
        # --- Sequential Accuracy (Top-1) ---
        is_m_seq = (torch.rand(half_B, cfg.seq_len, device=device) > cfg.alpha0)
        z0 = x_seq.clone()
        z0[is_m_seq] = cfg.mask_token_id
        logits_seq, ord_seq = model(z0, x=x_seq, diffusion_mode=False)
        logits_z = logits_seq[:, cfg.seq_len:, :]
        
        x_shuffled_seq = torch.gather(x_seq, 1, ord_seq)
        is_m_shuffled_seq = torch.gather(is_m_seq, 1, ord_seq)
        
        preds_seq = torch.argmax(logits_z, dim=-1)
        correct_seq = (preds_seq == x_shuffled_seq) & is_m_shuffled_seq
        acc_seq = correct_seq.sum().float() / is_m_shuffled_seq.sum().clamp(min=1)
        top1_accs.append(acc_seq.item())
        
        # --- Diffusion Mask Reconstruction Rate ---
        alpha_t = cfg.alpha0 * 0.5
        is_m_diff = (torch.rand(half_B, cfg.seq_len, device=device) > alpha_t)
        zt = x_diff.clone()
        zt[is_m_diff] = cfg.mask_token_id
        logits_diff, ord_diff = model(zt, diffusion_mode=True)
        
        x_shuffled_diff = torch.gather(x_diff, 1, ord_diff)
        is_m_shuffled_diff = torch.gather(is_m_diff, 1, ord_diff)
        
        preds_diff = torch.argmax(logits_diff, dim=-1)
        correct_diff = (preds_diff == x_shuffled_diff) & is_m_shuffled_diff
        recon_rate = correct_diff.sum().float() / is_m_shuffled_diff.sum().clamp(min=1)
        mask_recon_rates.append(recon_rate.item())

    # --- Inference Throughput ---
    print("Measuring inference throughput...")
    dummy_input = torch.full((1, cfg.seq_len), cfg.mask_token_id, device=device)
    for _ in range(5):
        _ = model(dummy_input, diffusion_mode=True)
        
    start_time = time.time()
    num_runs = 50
    total_tokens = num_runs * cfg.seq_len
    for _ in range(num_runs):
        _ = model(dummy_input, diffusion_mode=True)
    end_time = time.time()
    
    tps = total_tokens / (end_time - start_time)
    
    print("\n" + "="*30)
    print("      EVALUATION RESULTS")
    print("="*30)
    print(f"Top-1 Accuracy (AR):     {sum(top1_accs)/len(top1_accs):.2%}")
    print(f"Mask Recon Rate (Diff):  {sum(mask_recon_rates)/len(mask_recon_rates):.2%}")
    print(f"Inference Throughput:    {tps:.2f} tokens/sec")
    print("="*30)

if __name__ == "__main__":
    run_evaluation()
