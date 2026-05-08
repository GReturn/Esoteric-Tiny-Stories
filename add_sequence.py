import json

with open('esolm_faithful_roadmap.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def generate_diffusion(' in source:
            new_source = """@torch.no_grad()
def generate_diffusion(
    prompt,
    steps=32,
    temperature=1.0,
    show_steps=False
):

    model.eval()

    x = torch.tensor(
        [encode(prompt)],
        device=device
    )

    B, L0 = x.shape
    gen_len = cfg.seq_len - L0

    mask_tokens = torch.full(
        (1, gen_len),
        cfg.mask_token_id,
        device=device
    )

    x = torch.cat([x, mask_tokens], dim=1)
    visible = x != cfg.mask_token_id

    if show_steps:
        print(f"Init : {decode(x[0].tolist())}")

    for step in range(steps):
        logits = model(x, visible)
        probs = F.softmax(
            logits / temperature,
            dim=-1
        )
        sampled = torch.distributions.Categorical(probs).sample()
        confidence = probs.max(dim=-1).values

        masked_positions = (x == cfg.mask_token_id)
        x[masked_positions] = sampled[masked_positions]

        remaining_ratio = 1 - (step + 1) / steps
        k = int(remaining_ratio * masked_positions.sum().item())

        if k > 0:
            masked_conf = confidence.clone()
            masked_conf[~masked_positions] = 999
            flat = masked_conf.view(-1)
            idx = torch.topk(
                flat,
                k,
                largest=False
            ).indices
            flat_x = x.view(-1)
            flat_x[idx] = cfg.mask_token_id
            x = flat_x.view_as(x)

        visible = x != cfg.mask_token_id
        
        if show_steps:
            print(f"Step {step+1:02d}: {decode(x[0].tolist())}")

    return decode(x[0].tolist())"""
            cell['source'] = [line + "\n" for line in new_source.split("\n")]

        if 'print(generate_diffusion("ROMEO:' in source:
            cell['source'] = ['print(generate_diffusion("ROMEO:\\n", steps=10, show_steps=True))\n']

with open('esolm_faithful_roadmap.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
