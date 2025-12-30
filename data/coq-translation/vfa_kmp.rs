use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: KMP String Matching (Supporting VFA)
// ============================================================================

// Compute failure function for KMP
pub open spec fn compute_failure(pattern: Seq<nat>, i: nat, j: nat) -> Seq<nat>
    decreases pattern.len() - i
{
    if i >= pattern.len() { Seq::empty() }
    else {
        let new_j = if j > 0 && pattern[i as int] == pattern[j as int] { j + 1 } else { 0 };
        seq![new_j] + compute_failure(pattern, i + 1, new_j)
    }
}

pub open spec fn failure_table(pattern: Seq<nat>) -> Seq<nat> {
    seq![0nat] + compute_failure(pattern, 1, 0)
}

// Simplified KMP search with fuel for termination
pub open spec fn kmp_match_fuel(text: Seq<nat>, pattern: Seq<nat>, ti: nat, pi: nat, fail: Seq<nat>, fuel: nat) -> Option<nat>
    decreases fuel
{
    if fuel == 0 { None }
    else if pattern.len() == 0 { Some(0) }
    else if ti >= text.len() { None }
    else if pi >= pattern.len() { Some((ti - pi) as nat) }
    else if ti < text.len() && pi < pattern.len() && text[ti as int] == pattern[pi as int] {
        kmp_match_fuel(text, pattern, ti + 1, pi + 1, fail, (fuel - 1) as nat)
    }
    else if pi > 0 && pi <= fail.len() {
        let new_pi = if (pi - 1) < fail.len() { fail[(pi - 1) as int] } else { 0 };
        kmp_match_fuel(text, pattern, ti, new_pi, fail, (fuel - 1) as nat)
    }
    else {
        kmp_match_fuel(text, pattern, ti + 1, 0, fail, (fuel - 1) as nat)
    }
}

pub open spec fn kmp_match(text: Seq<nat>, pattern: Seq<nat>) -> Option<nat> {
    let fail = failure_table(pattern);
    kmp_match_fuel(text, pattern, 0, 0, fail, text.len() + pattern.len())
}

// Properties
pub proof fn failure_table_length(pattern: Seq<nat>)
    ensures failure_table(pattern).len() >= 1
{
}

pub proof fn example_kmp()
{
    reveal_with_fuel(kmp_match_fuel, 3);
    reveal_with_fuel(compute_failure, 3);

    // Empty pattern matches at position 0
    let text = seq![1nat, 2, 3];
    let pattern = Seq::<nat>::empty();
    assert(kmp_match(text, pattern) == Some(0nat));
}

pub proof fn kmp_verify() {
    example_kmp();
}

pub fn main() {
    proof {
        kmp_verify();
    }
}

} // verus!
