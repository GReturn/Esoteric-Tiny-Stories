import torch
probs = torch.tensor([[0.5, float('nan'), 0.5]], device='cuda')
print("Testing multinomial with NaN")
try:
    sampled = torch.multinomial(probs, 1)
    print("Success:", sampled)
except Exception as e:
    print(f"Error: {e}")
