# Dataset Creation Proposal for VerusSFT

## Goal
Create a high-quality, dependency-clean dataset of self-contained Rust/Verus code snippets—each of which compiles and verifies independently—to support supervised fine-tuning (SFT) and reinforcement learning (RL) for formal verification tasks. Each example will be:

- Tagged as **code**, **spec**, or **proof**
- Validated by Verus for successful or failing verification
- Linked to its source provenance (repo, file path, commit SHA)
- Serialized in a machine-readable format (e.g., JSONL) for use in downstream tasks

This dataset will seed all SFT and RL experiments in VerusSFT and support benchmarking and curriculum design.

## Proposed Methods
We will explore three complementary directions for dataset construction. A robust dataset will likely combine outputs from all three.

### 1. Minimizer-Driven Extraction (Baseline Path)
- **Overview:** Use and extend Verus’s existing minimizer tool (based on `creduce`) to shrink verification programs while preserving successful or failing behavior.
- **Method:**
  - Collect full Verus source files from curated repos (e.g., `verus-lang/verus`, VeriStruct modules, Verismo, etc.).
  - Apply the Verus minimizer with an "interestingness test" (e.g., `verus --verify` with success/failure checks).
  - Wrap output in a standalone crate and validate it compiles/verifies.
  - Label and segment each snippet into exec, spec, and proof zones.
  - Track metadata: origin repo, commit SHA, minimized status, verification result.
- **Strengths:** Leverages existing Verus infrastructure; produces semantically minimal, verification-relevant examples; aligns with RL trajectory mining and failure analysis.
- **Challenges:** Minimization can be slow; some outputs may be over-fragmented or hard to read; might miss human-realistic examples.
- **Status:** Implementation-ready; favored as the first dataset extraction path.

### 2. Program-Slicing + Verus IR-Aware Extraction
- **Overview:** Use static analysis tools (e.g., Flowistry, compiler instrumentation) to slice Rust/Verus programs into smaller, role-tagged units based on control/data dependencies and Verus IR annotations.
- **Method:**
  - Parse AST or intermediate representations (IR) of Verus programs.
  - Identify functions, lemmas, view functions, invariants, etc.
  - Slice programs to retain only the minimal set of dependencies for a particular construct.
  - Use the Verus mode system (exec, ghost, proof) to tag output.
  - Validate that each extracted unit compiles/verifies independently.
- **Strengths:** Produces semantically meaningful slices (e.g., whole lemmas or proofs); avoids over-fragmentation; can track logical dependency graphs for curriculum design.
- **Challenges:** Flowistry and other Rust tools may not be Verus-compatible; requires deeper instrumentation or modification of Verus compiler internals; likely high engineering cost initially.
- **Status:** High-reward but experimental; to be explored after the baseline dataset is collected.

### 3. AI-Assisted Example Generation and Refinement
- **Overview:** Use a language model (LLM) to propose or refine candidate snippets, with Verus-in-the-loop feedback to ensure dependency-clean, verifiable outputs.
- **Method:**
  - Seed the LLM with code/spec/proof templates from known examples.
  - Prompt the LLM to generate variations, trim dependencies, or repair broken examples.
  - Run Verus verification as feedback; refine or reject based on outcomes.
  - Use the model to suggest human-like completions for partially minimized examples, repair broken code based on error messages, and denoise or simplify multi-function modules.
- **Strengths:** Can generate realistic, semantically rich training examples; useful for hard-to-minimize edge cases (e.g., interactive proof patterns); directly aligned with VerusSFT's long-term RL and agent training goals.
- **Challenges:** Low precision without strong model priors (especially on ghost/view logic); requires verifier-in-the-loop and filtering; may introduce hallucinated or non-compiling artifacts if not constrained.
- **Status:** High-potential augmentation strategy, to be applied selectively after initial dataset bootstrapping.

## Common Requirements Across Methods
- Automate compilation and verification for every snippet to ensure dataset reliability.
- Track provenance (original file paths, commit hashes) for traceability.
- Provide machine-readable labels and metadata to support downstream SFT workflows.
