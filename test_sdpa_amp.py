import torch
import torch.nn.functional as F

device = torch.device('cuda')

def get_esolm_b_diffusion_mask(is_masked: torch.Tensor):
    B, T = is_masked.shape
    device = is_masked.device
    rand_keys = torch.rand(B, T, device=device)
    sort_keys = rand_keys + is_masked.float() 
    sigma_inv = torch.argsort(torch.argsort(sort_keys, dim=1), dim=1) 
    sigma_inv_q = sigma_inv.unsqueeze(2) 
    sigma_inv_k = sigma_inv.unsqueeze(1) 
    allowed = sigma_inv_q >= sigma_inv_k 
    
    attn_bias = torch.zeros(B, T, T, device=device)
    attn_bias.masked_fill_(~allowed, -10000.0)
    return attn_bias.unsqueeze(1)

B, n_head, T, head_dim = 1, 6, 256, 64
q = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.float16)
k = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.float16)
v = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.float16)

is_masked = torch.ones(B, T, dtype=torch.bool, device=device)
is_masked[0, :10] = False

attn_bias = get_esolm_b_diffusion_mask(is_masked)

print("Testing float16 with -10000.0")
try:
    with torch.autocast('cuda'):
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias.to(q.dtype),
            is_causal=False, 
        )
    print("Success")
except Exception as e:
    print(f"Error: {e}")

