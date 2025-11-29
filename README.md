# VerusFT-RL: Fine-Tuning and RL for Verification-Oriented Rust/Verus Code

VerusFT-RL explores supervised fine-tuning (SFT) and reinforcement learning (RL) for language models that need to reason about verification-oriented Rust/Verus code. The repo currently contains a small prototype SFT pipeline (GPT-2 + LoRA + a handful of Verus examples) and is evolving into a full dataset builder, task-specific training suite, and reproducible benchmark for verification-aware code generation.

## Overview

General-purpose code LLMs often struggle with Verus-specific concepts such as `exec`/`ghost`/`proof` modes, `requires`/`ensures` specifications, View functions, typestate-like abstractions, loop invariants and `decreases` clauses, and proof blocks. VerusFT-RL aims to make models genuinely Verus-aware and to understand when structured representations (like ASTs) add value beyond plain text.

## Project Goals

### Primary goals
1. **Build a minimized, high-quality text-only dataset** of Verus code, specs, proofs, and error traces from open-source projects.
2. **Train SFT models** on three core tasks:
   - **Task A — Code → Specifications** (generate `requires`/`ensures`, Views, invariants).
   - **Task B — Specifications → Verified Code** (fill in executable + ghost + proof code).
   - **Task C — Error-Guided Proof/Invariant Repair** (fix code/specs using Verus error messages).
3. **Evaluate models using Verus itself** (verification pass rate), not just syntax or text similarity.

### Secondary goal (ablation / bonus)
4. After strong text-only baselines exist, introduce **AST/structured encodings** and measure their incremental benefit via ablation studies.

## Motivation

Formal verification embeds specifications, invariants, and proofs directly in Rust code. Early assistants show LLMs can help, but current models often mishandle modes, omit specs, generate invalid proofs, or misread Verus error messages. Our hypothesis: supervised fine-tuning on a curated corpus of minimized, self-contained Verus examples will significantly improve performance on spec generation, verified code synthesis, and proof repair—even without structured encodings. With strong text baselines in place, we can rigorously test when ASTs or other structural views truly help.

## Features

- ✅ Parameter-efficient fine-tuning with LoRA
- ✅ 10 diverse Verus training examples (seed set, expanding soon)
- ✅ Configurable training hyperparameters
- ✅ Inference script for testing trained models
- ✅ Small adapter weights (~6.2MB) instead of full model

## Methodology

The project is split into two main phases.

### Phase 1: Text-only dataset + SFT (core)
- Start from existing open-source Verus code (examples, tests, verified libraries, and projects listed on the Verus publications/projects page).
- Use the Verus minimizer ([`source/tools/minimizers/`](https://github.com/ChuyueSun/verus/tree/main/source/tools/minimizers)) to shrink large programs into tiny, self-contained examples that preserve a chosen verification behavior.
- Extract logical units from minimized files:
  - Function-level units for **Task A (code → spec)** and **Task B (spec → code)**.
  - Lemma/proof-level units for **Task C (repair)**.
  - Module-level units only when necessary (e.g., shared View functions).
- Serialize each dataset entry as text-only JSONL with metadata (task label, minimized flag, Verus version, code size).
- Train SFT models on Tasks A/B/C and evaluate with Verus-based pass rates.

### Phase 2: AST / structured representation ablation (bonus)
- Add structured views (linearized Rust ASTs, control-flow summaries, verification-state descriptors, proof-tree or lemma-call graphs).
- Compare **text-only**, **structure-only**, and **text + structure** across tasks to quantify when structure helps (e.g., nested loops, complex invariants, higher-order lemmas).

### RL strategy
- Begin with **offline RL** to control Verus evaluation cost: log every agent interaction (code snapshots, tool calls, Verus logs), label/filter trajectories, train offline policies (e.g., Decision Transformers), and validate in a replay simulator.
- Transition to **online PPO/GRPO or reflection loops** after offline evaluations look solid, using the offline policy as initialization.

## Training results (prototype)

- **Loss reduction**: 3.40 → 2.42 (28.7% improvement)
- **Token accuracy**: 45% → 55%
- **Training time**: ~14 seconds for 8 epochs
- **Adapter size**: 6.2MB

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd VerusFT-RL

# Install dependencies
pip install transformers trl datasets peft accelerate torch
```

## Usage

### 1. Training

Run the training script to fine-tune the model on Verus examples:

```bash
python sft_example.py
```

The trained model will be saved to `./sft_output/`.

### 2. Inference

Test the trained model with new prompts:

```bash
python test_inference.py
```

## Configuration

### Training parameters
- `num_train_epochs`: Number of training epochs (default: 10)
- `per_device_train_batch_size`: Batch size per device (default: 2)
- `learning_rate`: Learning rate (default: 3e-4)
- `max_seq_length`: Maximum sequence length (default: 1024)

### LoRA configuration
- `r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA alpha parameter (default: 32)
- `target_modules`: Modules to apply LoRA (default: `["c_attn", "c_proj"]`)

## Dataset

The seed training dataset includes 10 examples covering:
- Absolute value functions
- Min/max operations
- Arithmetic operations (add, subtract, multiply, divide)
- Array bounds checking
- Boolean predicates
- Squaring with overflow protection

### Adding more examples

Edit the `build_dataset()` function in `sft_example.py` to add your own Verus examples:

```python
{
    "text": "Your prompt here\nYour Verus code here"
}
```

Possible sources of additional Verus examples include:
- [VOSTD dataset](https://github.com/asterinas/vostd) for automatically generated specifications
- [Verismo](https://github.com/microsoft/verismo) for verified systems examples
- [Verified Memory Allocator](https://github.com/verus-lang/verified-memory-allocator) for memory-safety focused code
- [Verified Storage](https://github.com/microsoft/verified-storage) for storage system verification examples
- [Vericoding](https://github.com/Beneficial-AI-Foundation/vericoding) for related work and potential benchmarking ideas

## Student subprojects

This repo is designed to support multiple small research projects (e.g., rotation or undergraduate projects). Example subprojects:
1. **Dataset via minimizer (core plumbing)**: Script the minimizer calls; build JSONL datasets for Tasks A/B/C; implement deduplication and quality filters.
2. **SFT for spec generation (Task A)**: Train models on code → spec; evaluate on held-out modules.
3. **SFT for verified code synthesis (Task B)**: Train models on spec → code; evaluate by running Verus on generations.
4. **SFT for proof/invariant repair (Task C)**: Build a dataset of (broken example, error) → (fixed example); train models to repair common errors.
5. **Benchmark & evaluation harness**: Automate Verus compilation, runs, and metric collection.
6. **AST / structure ablation study (advanced)**: Design structured encodings and run controlled ablations vs. text-only baselines.

## File structure

```
VerusFT-RL/
├── sft_example.py          # Main training script
├── test_inference.py       # Inference/testing script
├── README.md               # Combined overview + proposal
├── .gitignore              # Git ignore rules
└── sft_output/             # Trained model output (not tracked)
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── tokenizer files
```

## Future improvements

1. **Expand dataset**: Add 50–100+ diverse Verus examples.
2. **Better base model**: Try code-specific models such as `Qwen/Qwen2.5-Coder-1.5B`, `bigcode/starcoder2-3b`, or `deepseek-ai/deepseek-coder-1.3b-base`.
3. **Evaluation**: Add metrics for Verus specification correctness and verification pass rate.
4. **Fine-tune generation**: Adjust decoding parameters (temperature, top-p, beam search) and RL reward shaping.
5. **Structured ablations**: Quantify when AST or other structured signals meaningfully improve verification outcomes.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- trl
- datasets
- peft
- accelerate

## License

MIT License

## Acknowledgments

- [Verus](https://github.com/verus-lang/verus) - The Verus verification system
- [Hugging Face](https://huggingface.co/) - Transformers and model hub
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning library
