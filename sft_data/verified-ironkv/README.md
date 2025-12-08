# Verus SFT Dataset - IronKV

This directory contains a curated dataset for training models on Verus verification tasks, extracted from the IronKV codebase.

## Dataset Overview

**Total: 514 training examples** across three verification tasks:

| Task | Description | Examples | Files to Use |
|------|-------------|----------|--------------|
| **Task A** | Signature + Body → Specifications | 173 | `task_a_*_fixed.jsonl` |
| **Task B** | Specifications → Implementation | 159 | `task_b_*.jsonl` |
| **Task C** | Error + Broken Code → Fixed Code | 182 | `task_c_*_filtered.jsonl` |

**Splits:**
- Training: 410 examples (138 + 127 + 145)
- Validation: 50 examples (17 + 15 + 18)
- Test: 54 examples (18 + 17 + 19)

## Quick Start

### View the Dataset

```bash
# Browse examples interactively
python view_dataset.py raw/task_a_train_fixed.jsonl
python view_dataset.py raw/task_b_train.jsonl
python view_dataset.py raw/task_c_train_filtered.jsonl

# Or view in browser
open html_reports/index.html
```

### Edit the Dataset

```bash
# Interactive editor - browse and edit entries
python edit_dataset.py raw/task_a_train_fixed.jsonl

# Press 'e' to edit, 's' to save, 'q' to quit
```

### Files for Training

**Recommended files (use these):**

```bash
# OpenAI format
openai_format/task_a_train_openai_fixed.jsonl       # 138 examples
openai_format/task_b_train_openai.jsonl              # 127 examples
openai_format/task_c_train_openai_filtered.jsonl    # 145 examples

# ShareGPT format
sharegpt_format/task_a_train_sharegpt_fixed.jsonl   # 138 examples
sharegpt_format/task_b_train_sharegpt.jsonl          # 127 examples
sharegpt_format/task_c_train_sharegpt_filtered.jsonl # 145 examples

# Raw format (for custom pipelines)
raw/task_a_train_fixed.jsonl                         # 138 examples
raw/task_b_train.jsonl                               # 127 examples
raw/task_c_train_filtered.jsonl                      # 145 examples
```

## Task Definitions

### Task A: Specification Inference

**Input:** Function signature + implementation body (without specs)
**Output:** Specifications (requires, ensures, invariants, decreases)

**Example:**
```
Input:
  pub fn clone_option_end_point(oep: &Option<EndPoint>) -> (cloned_oep: Option<EndPoint>)
  {
      match oep.as_ref() {
          Some(ep) => Some(clone_end_point(ep)),
          None => None
      }
  }

Output:
  ensures match oep {
      Some(ep) => cloned_oep.is_some() && ep@ == cloned_oep->0@,
      None => cloned_oep is None,
  },
```

**Files:** Use `task_a_*_fixed.jsonl` (corrected to include function body in input)

### Task B: Code Generation from Specifications

**Input:** Function signature + specifications
**Output:** Verified implementation body (executable + ghost + proof code)

**Example:**
```
Input:
  Specifications:
  pub fn get(&self, key: &K) -> (result: Option<&V>)
    requires self.valid()
    ensures match result {
        Some(v) => self@.contains_key(*key) && self@[*key] == v@,
        None => !self@.dom().contains(*key),
    }

Output:
  {
      match self.m.get(&key) {
          Some(v) => Some(v),
          None => None,
      }
  }
```

**Files:** Use original `task_b_*.jsonl` files

### Task C: Error-Guided Repair

**Input:** Code with verification error + error message
**Output:** Fixed code that verifies

**Example:**
```
Input:
  Error: Verification failed after remove_requires

  Broken Code:
  pub fn insert(&mut self, key: K, value: V)
  {
      self.m.insert(key, value);
  }

Output:
  pub fn insert(&mut self, key: K, value: V)
    requires old(self).valid()
    ensures self.valid(), self@ == old(self)@.insert(key, value@)
  {
      self.m.insert(key, value);
  }
```

**Files:** Use `task_c_*_filtered.jsonl` (80 problematic entries removed)

## Directory Structure

```
sft_data/verified-ironkv/
├── raw/                              # Raw JSONL format
│   ├── task_a_*_fixed.jsonl         # Task A with function bodies (USE THESE)
│   ├── task_b_*.jsonl               # Task B (original is correct)
│   ├── task_c_*_filtered.jsonl      # Task C filtered (USE THESE)
│   └── *.jsonl.backup               # Automatic backups
│
├── openai_format/                    # OpenAI chat format
│   ├── task_a_*_openai_fixed.jsonl
│   ├── task_b_*_openai.jsonl
│   └── task_c_*_openai_filtered.jsonl
│
├── sharegpt_format/                  # ShareGPT conversation format
│   ├── task_a_*_sharegpt_fixed.jsonl
│   ├── task_b_*_sharegpt.jsonl
│   └── task_c_*_sharegpt_filtered.jsonl
│
├── html_reports/                     # Interactive HTML viewers
│   ├── index.html                   # Main entry point
│   ├── task_a_report.html
│   ├── task_b_report.html
│   └── task_c_filtered_report.html
│
├── text_samples/                     # Plain text samples
│   ├── task_a_sample.txt
│   ├── task_b_sample.txt
│   └── task_c_filtered_sample.txt
│
├── view_dataset.py                   # Interactive viewer
├── edit_dataset.py                   # Interactive editor
├── check_data_quality.py             # Quality validation
├── analyze_specific_issues.py        # Detailed analysis
├── filter_task_c.py                  # Filter Task C duplicates
├── fix_task_a.py                     # Fix Task A to include bodies
│
├── DATA_QUALITY_REPORT.md            # Comprehensive quality analysis
├── QUICK_SUMMARY.md                  # At-a-glance summary
├── FIXED_DATASET_README.md           # Explanation of fixes
├── VIEWING_GUIDE.md                  # How to view the data
├── EDITING_GUIDE.md                  # How to edit the data
└── README.md                         # This file
```

## Tools

### View Data

```bash
# Interactive viewer (browse with n/p/j/s/f/q)
python view_dataset.py <jsonl_file>

# HTML reports (open in browser)
open html_reports/index.html

# Simple shell script
./browse_dataset.sh <jsonl_file>

# Generate new reports
python generate_html_reports.py
python generate_text_samples.py
```

### Edit Data

```bash
# Interactive editor (browse and edit with e/d/s/q)
python edit_dataset.py <jsonl_file>

# Automatically creates backup on first edit
# Press 'e' to edit current entry in your text editor
# Press 's' to save changes
```

### Quality Checks

```bash
# Run comprehensive quality validation
python check_data_quality.py

# Detailed issue analysis
python analyze_specific_issues.py

# Filter Task C duplicates
python filter_task_c.py

# Fix Task A to include function bodies
python fix_task_a.py
```

## Data Quality

**Task A (Fixed):**
- ✅ 173 examples with function bodies in input
- ✅ No duplicates
- ✅ Average complexity: 7.6
- ⚠️ 2 examples with TODO comments (from source code)

**Task B (Original):**
- ✅ 159 examples
- ✅ No duplicates
- ✅ Average complexity: 8.1
- ⚠️ 16 examples with TODO comments (from source code)

**Task C (Filtered):**
- ✅ 182 clean examples (80 removed)
- ✅ No identical broken/fixed code
- ✅ Error types: 62% postcondition, 30% precondition, 8% invariant
- ❌ Original had 80 entries with identical broken/fixed code (now filtered)

**Overall Grade: A- (after fixes and filtering)**

See [DATA_QUALITY_REPORT.md](./DATA_QUALITY_REPORT.md) for detailed analysis.

## Important Notes

### Task A Fixed

⚠️ **The original Task A data was incorrect** - it only had function signatures without bodies, making it impossible to infer specifications.

**Use the fixed versions:**
- `task_a_*_fixed.jsonl` (includes function bodies)
- `task_a_*_openai_fixed.jsonl`
- `task_a_*_sharegpt_fixed.jsonl`

See [FIXED_DATASET_README.md](./FIXED_DATASET_README.md) for details.

### Task C Filtered

⚠️ **The original Task C had 80 entries with identical broken/fixed code** - these won't teach the model anything.

**Use the filtered versions:**
- `task_c_*_filtered.jsonl` (182 clean examples)
- `task_c_*_openai_filtered.jsonl`
- `task_c_*_sharegpt_filtered.jsonl`

## Data Formats

### Raw Format

```json
{
  "id": "task_a_example_123",
  "task": "code_to_spec",
  "input_text": "Given the following Verus function implementation...",
  "target_text": "ensures ...",
  "full_function": "pub fn example() { ... }",
  "metadata": {
    "source_file": "../ironsht/src/example.rs",
    "function_name": "example",
    "function_mode": "exec",
    "complexity_score": 5,
    ...
  }
}
```

### OpenAI Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in Verus..."
    },
    {
      "role": "user",
      "content": "Given the following Verus function implementation..."
    },
    {
      "role": "assistant",
      "content": "ensures ..."
    }
  ]
}
```

### ShareGPT Format

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are an expert in Verus..."
    },
    {
      "from": "human",
      "value": "Given the following Verus function implementation..."
    },
    {
      "from": "gpt",
      "value": "ensures ..."
    }
  ]
}
```

## Training Recommendations

1. **Start with Task A and B** - cleaner data with higher quality
2. **Use filtered Task C** - avoid the 80 problematic entries
3. **Use fixed Task A files** - includes function bodies in input
4. **Validation sets are small** (15-18 examples) - consider combining or using more data

**Total usable training data: 410 examples**
- This is relatively small for LLM fine-tuning
- Consider data augmentation or combining with other Verus datasets
- May work well for instruction tuning on top of a code model

## Source Attribution

This dataset was extracted from the IronKV codebase:
- Source: `../ironsht/src/`
- All code is from verified Verus implementations
- Functions include specifications, proofs, and ghost code

## License

The dataset inherits the license from the IronKV source code.

## Questions?

See the documentation:
- [QUICK_SUMMARY.md](./QUICK_SUMMARY.md) - Overview and key findings
- [DATA_QUALITY_REPORT.md](./DATA_QUALITY_REPORT.md) - Detailed quality analysis
- [FIXED_DATASET_README.md](./FIXED_DATASET_README.md) - What was fixed and why
- [VIEWING_GUIDE.md](./VIEWING_GUIDE.md) - How to view the data
- [EDITING_GUIDE.md](./EDITING_GUIDE.md) - How to edit the data
