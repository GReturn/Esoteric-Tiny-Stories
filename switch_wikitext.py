import json

with open('esolm_faithful_roadmap.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if "shakespeare.txt" in line:
                cell['source'][i] = line.replace("shakespeare.txt", "wikitext2.txt")
            if "shakespeare_bpe" in line:
                cell['source'][i] = line.replace("shakespeare_bpe", "wikitext_bpe")
            if "k = int(remaining_ratio * masked_positions.sum().item())" in line:
                cell['source'][i] = line.replace("masked_positions.sum().item()", "gen_len")

with open('esolm_faithful_roadmap.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
