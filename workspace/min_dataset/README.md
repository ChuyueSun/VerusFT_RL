# Verus Formal Verification Dataset

Training dataset for Verus formal verification tasks, containing **14,659 entries** across three task types designed to teach models specification generation, verified code synthesis, and proof repair.

**Location:** `workspace/min_dataset/`
```
min_dataset/
├── dataset.jsonl              # Complete dataset (14,659 entries)
├── dataset.sqlite3            # SQLite database for queries (this is just for easier querying of the dataset, like through `Harlequin`)
├── stats.json                 # Dataset statistics
└── splits/
    ├── train.jsonl            # Combined train (11,725 entries, 80%)
    ├── val.jsonl              # Combined val (1,466 entries, 10%)
    ├── test.jsonl             # Combined test (1,468 entries, 10%)
    ├── task_a_train.jsonl     # Task A train (2,941)
    ├── task_a_val.jsonl       # Task A val (368)
    ├── task_a_test.jsonl      # Task A test (368)
    ├── task_b_train.jsonl     # Task B train (2,600)
    ├── task_b_val.jsonl       # Task B val (325)
    ├── task_b_test.jsonl      # Task B test (326)
    ├── task_c_train.jsonl     # Task C train (6,184)
    ├── task_c_val.jsonl       # Task C val (773)
    ├── task_c_test.jsonl      # Task C test (774)
    └── metadata.json          # Split metadata
```

**JSONL Format:**
```json
{
  "id": "task_a_7b5f7e94f536",
  "task": "task_a",
  "input_text": "fn findMax(a: &[i32]) -> (max: i32) { ... }",
  "target_text": "requires a.len() > 0\nensures forall|k: int| ...",
  "full_verified_code": "<complete verified source>",
  "source": "vericoding",
  "source_file": "findMax.rs",
  "verified": true,
  "metadata": { "bug_type": null }
}
```

**Training usage:**
```python
import json

with open("splits/train.jsonl") as f:
    train_data = [json.loads(line) for line in f]

# Filter by task type
task_a_train = [e for e in train_data if e["task"] == "task_a"]
task_b_train = [e for e in train_data if e["task"] == "task_b"]
task_c_train = [e for e in train_data if e["task"] == "task_c"]
```

---
## Construction

The dataset was constructed by extracting verified functions from multiple Verus sources and generating training tasks through AST-based transformations. Three extraction pipelines were developed: a high-yield pipeline for `vericoding-benchmark` (9,079 self-contained samples), a Coq→Verus translation pipeline for Software Foundations proofs (4,985 samples from 222 verified files), and a minimizer-driven pipeline for complex multi-file repositories (595 samples).

Source repositories include `vericoding-benchmark` (LLM-generated solutions to verification problems), `coq-translation` (Software Foundations Coq proofs translated to Verus, covering boolean logic, lists, Hoare logic, induction principles, and verified data structures), and nine complex repositories: `verus-lang/verus` (official examples), `vostd` (verified OS standard library), `verified-ironkv` (distributed key-value store), `verismo` (confidential VM monitor), `verified-storage` (persistent memory), `verified-memory-allocator`, `verified-node-replication`, and `verified-nrkernel`. ==The vericoding-benchmark provides ~62% of entries as self-contained exec functions, coq-translation adds ~34% as spec/proof functions, while complex repositories contribute ~4% due to expensive per-function minimization with low success rates.==

Verification was performed using a pre-built Verus binary (`verus-arm64-macos`) to ensure consistent results. Each extracted function was verified in isolation before inclusion. Functions containing `assume(false)`, `unimplemented!()`, or other verification shortcuts were excluded to ensure all training targets represent genuine verified code.

---
## Extraction Pipeline

The primary extractor (`verus_extractor/`) is a Rust tool built on `verus_syn`, the official Verus syntax parsing crate. This was chosen over regex-based extraction after discovering that string manipulation failed on edge cases: keywords appearing in comments, state machine macro definitions, and method calls like `.invariant()` being confused with loop invariants.

The extractor performs a full AST traversal using the `syn::visit` pattern. For each function, it collects:
- **Function-level specs:** `requires`, `ensures`, `decreases` clauses on the function signature
- **Loop-level specs:** `invariant` and `decreases` clauses inside `while`/`loop` constructs
- **Proof assertions:** `assert`, `assert_by`, `assert_forall_by` statements

```rust
struct AnnotationExtractor {
    annotations: ProofAnnotations,
}

impl<'ast> Visit<'ast> for AnnotationExtractor {
    fn visit_expr_while(&mut self, node: &'ast ExprWhile) {
        // Extract invariant [...] and decreases [...] from loop spec
    }
}
```

==The AST-based approach was critical for Task B correctness.== Early regex-based extraction leaked loop invariants into Task B inputs, which reference implementation variables (`idx`, `sum`, `max_val`) that don't exist in the function signature. The model would see specs mentioning undefined variables, making the task unsolvable. AST extraction cleanly separates function-level specs (safe to show) from loop-level specs (must hide).

---
## Task Types

### Task A: Code → Specs (3,677 entries)

The model receives a complete function implementation with all proof annotations stripped and must generate the full specification. This teaches inferring preconditions, postconditions, loop invariants, and termination measures from code behavior.

```rust
// INPUT: function with all specs removed
fn findMax(a: &[i32]) -> (max: i32) {
    let mut max = a[0];
    let mut i = 1;
    while i < a.len() {
        if a[i] > max { max = a[i]; }
        i += 1;
    }
    max
}

// TARGET: all proof annotations
requires a.len() > 0
ensures exists|k: int| 0 <= k < a.len() && max == a[k],
        forall|k: int| 0 <= k < a.len() ==> max >= a[k]
decreases a.len() - i
invariant 0 <= i <= a.len(),
          forall|k: int| 0 <= k < i ==> max >= a[k]
```

The input is generated by stripping all `requires`, `ensures`, `invariant`, `decreases`, and `assert` constructs from the verified code. ==49.3% of Task A entries include loop invariants in the target, and 46.6% include decreases clauses==, ensuring the model learns these harder annotation types.

### Task B: Specs → Code (3,251 entries)

The model receives a function signature with function-level specifications but no implementation and must synthesize code that satisfies the spec. This teaches verified code generation from formal specifications.

```rust
// INPUT: signature + function-level specs only
fn zap_negatives(a: &mut Vec<i32>)
    requires a.len() < 1000
    ensures a.len() == old(a).len(),
            forall|i: int| 0 <= i < a.len() ==>
                if old(a)[i] < 0 { a[i] == 0 } else { a[i] == old(a)[i] }

// TARGET: complete verified implementation
fn zap_negatives(a: &mut Vec<i32>)
    requires a.len() < 1000
    ensures ...
{
    let mut i = 0;
    while i < a.len()
        invariant i <= a.len(), ...
        decreases a.len() - i
    {
        if a[i] < 0 { a.set(i, 0); }
        i += 1;
    }
}
```

==Critically, Task B inputs exclude loop invariants and decreases clauses.== These reference implementation variables that only exist inside the function body. Including them would give the model "hints" about the implementation structure (loop variable names, bounds expressions) that shouldn't be available when synthesizing from specs alone.

### Task C: Repair (7,731 entries)

The model receives buggy code with one proof annotation removed and must identify and restore the missing annotation. This teaches debugging verification failures—a common task when working with formal verification tools.

```rust
// INPUT: code missing an invariant (verification fails)
fn sum_array(a: &Vec<i32>) -> (s: i32)
    requires a.len() < 100
    ensures s == spec_sum(a@)
{
    let mut s = 0;
    let mut i = 0;
    while i < a.len()
        // MISSING: invariant s == spec_sum(a@.take(i as int))
        decreases a.len() - i
    { ... }
}

// TARGET: complete verified code with invariant restored
```

Task C is generated by programmatically removing one annotation type from verified code. Five bug types are created:

| Bug Type | Count | Description |
|----------|-------|-------------|
| `missing_ensures` | 3,130 | Postcondition removed |
| `missing_requires` | 1,812 | Precondition removed |
| `missing_decreases` | 1,506 | Termination measure removed |
| `missing_invariant` | 1,010 | Loop invariant removed |
| `missing_assert` | 273 | Proof assertion removed |

---
## Minimization Approaches

### What Worked: Standalone Function Extraction

The vericoding-benchmark samples are ideal training data: single-file, self-contained functions averaging 60-80 lines. No minimization was needed. The extractor simply parsed each file, verified it, and generated task entries. This produced 2,055 high-quality samples with 89.8% verification success rate.

### What Partially Worked: AST-Based Function Isolation

For complex repositories, individual functions were extracted and verified in isolation. The `verus_syn` parser identified function boundaries, and each function was wrapped with necessary imports (`use vstd::prelude::*`) before standalone verification. This recovered 174 samples from complex repos (176 verification passes from 1,115 files scanned). ==The low yield (~16%) is due to inter-function dependencies: many functions call helpers, use custom types, or require module-level ghost state that can't be extracted standalone.==

### What Failed: C-Reduce Minimization

C-Reduce was attempted to shrink complex samples while preserving verification success. The tool is designed for bug-reproducing test cases, not training data preservation. Results were unusable:

```rust
// C-Reduce output (technically "verifies" with 0 functions)
use vstd;

// Or garbled when requiring ≥1 function:
fn d(a: Vec<int>) -> Vec<int> requires { Vec::new() }
```

Function names reduced to single letters, specs became empty or malformed, and formatting was destroyed. The tool optimizes for minimal reproduction, not meaningful code structure.

### What Was Abandoned: Whole-Crate Verification

Repositories like `vostd` and `verismo` require their full build context (`cargo xtask bootstrap`, custom toolchains, Linux-specific dependencies). Standalone Verus invocation fails immediately on any sample using `use crate::...` or `use super::...`. ==Approximately 80% of complex repository code falls into this category==, explaining the low extraction yield.

---
## Key Design Decisions

**AST-based annotation extraction over regex:** Regex patterns like `requires\s+(.*)` fail on multi-line clauses, nested expressions, and keywords in comments. The `verus_syn` crate provides reliable parsing.

**Separate function-level and loop-level specs:** Task B must not reveal implementation details. Loop invariants mentioning `idx`, `sum`, etc. would leak variable names to the model. The extractor distinguishes these at the AST level.

**Multiple bug types for Task C:** Rather than only removing `ensures` clauses, all five annotation types are candidates. This teaches the model to diagnose different verification failure modes.

**Filtering edge cases programmatically:** 13 entries contained keywords in comments or state machine macros (e.g., `invariant` in `tokenized_state_machine!`). Rather than re-running extraction, these were filtered post-hoc with targeted validation.

**80/10/10 split with fixed seed:** Reproducibility via `random.seed(42)`. The split is performed per-task-type to ensure balanced representation in each partition.

---
## Distribution

The dataset draws from three source categories:

| Source | Samples | % of Total | Description |
|--------|---------|------------|-------------|
| vericoding_ast | 9,079 | 62% | LLM-generated exec functions |
| coq_translation | 4,985 | 34% | Software Foundations spec/proof functions |
| complex_repos (verus, vostd, etc.) | 595 | 4% | Real Verus projects |

Within Task C, the distribution reflects annotation frequency: `ensures` clauses are most common (3,130), followed by `requires` (1,812), `decreases` (1,506), `invariant` (1,010), with `assert` statements least common (273).

---
## Split Strategy

The dataset is split 80/10/10 (train/val/test) with a fixed random seed (42) for reproducibility. Splits are performed independently per task type, then combined:

| Task | Train | Val | Test | Total |
|------|-------|-----|------|-------|
| Task A | 2,941 | 368 | 368 | 3,677 |
| Task B | 2,600 | 325 | 326 | 3,251 |
| Task C | 6,184 | 773 | 774 | 7,731 |
| **Combined** | **11,725** | **1,466** | **1,468** | **14,659** |

Both combined splits (all tasks mixed) and task-specific splits are provided. Use combined for general Verus competency training, or task-specific for curriculum learning or specialized fine-tuning.

---
## Training Recommendations

**Curriculum approach:** Start with Task A (easiest—inferring specs from visible code), progress to Task C (medium—identifying missing annotations), finish with Task B (hardest—synthesizing code from specs alone).

**Task mixing:** For general Verus competency, train on combined splits. The task type is encoded in each entry's `task` field, allowing the model to learn task-specific patterns.

**Verification signal:** All targets are verified-correct Verus code. Unlike unverified training data, every target actually passes the Verus checker, providing a reliable learning signal.

**Input/output format:** The `input_text` field is the model input; `target_text` is the expected output. The `full_verified_code` field contains the complete verified source for reference or alternative training formulations.

---
## Validation

The final dataset passed comprehensive validation:

- **Task A:** No proof annotations in input (0 issues), all targets contain specs
- **Task B:** No loop invariants leaked to input (0 issues), all entries have function signatures
- **Task C:** All targets contain the annotation type specified by `bug_type`
- **Edge cases:** 13 entries with keywords in comments/macros were filtered out
- **Verification:** All underlying functions pass Verus verification

The validation script checks for:
1. Function signatures with `requires`/`ensures` in Task A inputs (should be 0)
2. `while`/`loop` blocks with `invariant` in Task A inputs (should be 0)
3. Loop-level specs in Task B inputs (should be 0)
4. Correct bug type annotation in Task C targets
