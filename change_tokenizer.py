import json

with open('esolm_faithful_roadmap.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell index 1 (execution_count 4) - imports
nb['cells'][1]['source'] = [
    "\n",
    "import math\n",
    "import random\n",
    "from dataclasses import dataclass\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import sentencepiece as spm\n",
    "\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(\"device:\", device)\n"
]

# Cell index 3 (execution_count 5) - config
nb['cells'][3]['source'] = [
    "\n",
    "@dataclass\n",
    "class Config:\n",
    "    vocab_size = 8000\n",
    "    seq_len = 128\n",
    "\n",
    "    n_layers = 6\n",
    "    n_heads = 6\n",
    "    dim = 384\n",
    "    dropout = 0.1\n",
    "\n",
    "    batch_size = 16\n",
    "    lr = 3e-4\n",
    "    max_steps = 10000\n",
    "\n",
    "    mask_token_id = 0\n",
    "    pad_token_id = 0\n",
    "\n",
    "cfg = Config()\n"
]

# Cell index 5 (execution_count 6) - dataset & tokenizer
nb['cells'][5]['source'] = [
    "\n",
    "import sentencepiece as spm\n",
    "import os\n",
    "\n",
    "# Train BPE Tokenizer\n",
    "spm.SentencePieceTrainer.train(\n",
    "    input='shakespeare.txt', \n",
    "    model_prefix='shakespeare_bpe', \n",
    "    vocab_size=cfg.vocab_size,\n",
    "    user_defined_symbols=['[MASK]'],\n",
    "    pad_id=3,\n",
    "    unk_id=0,\n",
    "    bos_id=1,\n",
    "    eos_id=2\n",
    ")\n",
    "\n",
    "# Load Tokenizer\n",
    "sp = spm.SentencePieceProcessor()\n",
    "sp.load('shakespeare_bpe.model')\n",
    "\n",
    "# Dynamically set mask token in config\n",
    "cfg.mask_token_id = sp.piece_to_id('[MASK]')\n",
    "cfg.pad_token_id = sp.pad_id()\n",
    "\n",
    "encode = lambda s: sp.encode_as_ids(s)\n",
    "decode = lambda x: sp.decode_ids(x)\n",
    "\n",
    "with open(\"shakespeare.txt\", \"r\", encoding=\"utf-8\") as f:\n",
    "    text = f.read()\n",
    "\n",
    "data = torch.tensor(encode(text), dtype=torch.long)\n",
    "\n",
    "train_split = int(0.9 * len(data))\n",
    "train_data = data[:train_split]\n",
    "val_data = data[train_split:]\n",
    "\n",
    "print(\"vocab size:\", sp.get_piece_size())\n",
    "print(\"train tokens:\", len(train_data))\n"
]

with open('esolm_faithful_roadmap.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
