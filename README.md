# Eso-LM: Bridging Autoregressive and Masked Diffusion LLMs

[![Paper](https://img.shields.io/badge/arXiv-2506.01928-B31B1B.svg)](https://arxiv.org/abs/2506.01928)
[![Model](https://img.shields.io/badge/Architecture-Eso--LM%20(B)-blue.svg)](#)
[![Hardware](https://img.shields.io/badge/VRAM-8GB%20Optimized-green.svg)](#)

A high-fidelity implementation of the **Eso-LM (Variant B)** architecture. This model fuses Autoregressive (AR) and Masked Diffusion Model (MDM) paradigms into a single transformer backbone, enabling both exact likelihood computation and efficient KV-cached inference.

---

## Architectural Faithfulness

This implementation is mathematically aligned with the **Eso-LM (B)** setup described in *"Esoteric Language Models: Bridging Autoregressive and Masked Diffusion LLMs"* (arXiv:2506.01928).

> [!IMPORTANT]
> **Key Innovation**: Unlike standard MDMs that use bidirectional attention, Eso-LM (B) utilizes **Causal Attention** for both diffusion and sequential phases. This allows the model to leverage standard KV-caching optimizations, achieving up to 65x faster inference than traditional diffusion models.

### 1. Unified Any-Order Loss
The model optimizes a joint objective function $\mathcal{L} = \mathcal{L}_{AR} + \mathcal{L}_{MDM}$, controlled by the interpolation factor $\alpha_0$.
- **Sequential Phase ($\mathcal{L}_{AR}$)**: Learns the next-token prediction in a 2L "Teacher Forcing" topology.
- **Diffusion Phase ($\mathcal{L}_{MDM}$)**: Learns to denoise globally masked sequences with a noise-aware NELBO weighting.

### 2. Attention Masking Strategy
The model employs two distinct attention modes to bridge the paradigms:
- **`diffusion_mode=True`**: Standard causal masking applied to a randomly shuffled token sequence. This implements the "Any-Order AR" property required for diffusion.
- **`diffusion_mode=False`**: A specialized **2L Teacher Forcing Mask**. The input is doubled ($[x_{shuffled}, z_{shuffled}]$), where the second half ($z$) attends to the clean ground-truth prefix ($x$) and its own history, forcing the model to learn structure from clean context.

### 3. Physical Shuffling & RoPE
Instead of complex attention bias matrices, we utilize **Physical Reordering**. Tokens and their **Rotary Positional Embeddings (RoPE)** are physically shuffled in memory before the transformer blocks. This allows the use of high-performance, hardware-native causal kernels while maintaining mathematical correctness for the Any-Order objective.

---

## Performance Optimizations

To enable training of this Hugh Mungus architecture on consumer hardware (8GB VRAM), several industrial optimizations were implemented:

*   **Shuffled-Domain Loss**: Instead of aligning model logits (a $B \times L \times V$ operation), we shuffle the training targets to match the physical token order. This drastically reduces VRAM overhead.
*   **Zero-Latency Data Engine**: A GPU-native vectorized engine that prepares batches in VRAM, eliminating CPU bottlenecks during the diffusion noise sampling.
*   **Low-Discrepancy Sampling**: Uses a jittered noise schedule for the diffusion phase to ensure more uniform coverage of the noise space during small training runs.

---

## Technical Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Backbone** | Transformer | 8 Layers, 8 Heads, 512-dim |
| **Activation** | SwiGLU | Modern SiLU-based expansion for stable gradients |
| **Context** | 256 Tokens | Optimized for TinyStories dataset |
| **Vocab** | 10,000 | BPE with Byte-Fallback (SentencePiece) |
| **Interpolation** | $\alpha_0 = 0.5$ | Balances AR and Diffusion training |
| **VRAM Target** | ~6.5 GB | Fully compatible with RTX 3060/4060 |

---

## Getting Started

### Training
The training script handles dataset streaming, BPE training, and the dual-objective training loop.
```bash
python train_esolm.py
```

### Inference
The inference engine supports two generation modes. By default, it uses a **4-step diffusion sketch** followed by iterative refinement.
```bash
python inference_esolm.py
```

---

## References
- Sahoo et al. (2025). [Esoteric Language Models (Eso-LMs)](https://arxiv.org/abs/2506.01928)
- Eldan & Li (2023). [TinyStories: How Small Can Language Models Be?](https://arxiv.org/abs/2305.07759)
