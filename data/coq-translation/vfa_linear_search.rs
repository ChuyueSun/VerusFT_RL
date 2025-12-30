use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Linear Search (Supporting VFA)
// ============================================================================

pub open spec fn linear_search(s: Seq<nat>, target: nat, i: nat) -> Option<nat>
    decreases s.len() - i
{
    if i >= s.len() { None }
    else if s[i as int] == target { Some(i) }
    else { linear_search(s, target, i + 1) }
}

pub open spec fn find(s: Seq<nat>, target: nat) -> Option<nat> { linear_search(s, target, 0) }

pub open spec fn contains(s: Seq<nat>, x: nat) -> bool { find(s, x).is_some() }

pub proof fn search_finds_target(s: Seq<nat>, target: nat, i: nat)
    ensures match linear_search(s, target, i) {
        Some(idx) => i <= idx < s.len() && s[idx as int] == target,
        None => forall|j: nat| i <= j < s.len() ==> s[j as int] != target,
    }
    decreases s.len() - i
{
    reveal_with_fuel(linear_search, 2);
    if i < s.len() && s[i as int] != target { search_finds_target(s, target, i + 1); }
}

pub proof fn example_linear() {
    reveal_with_fuel(linear_search, 8);
    let s = seq![3nat, 1, 4, 1, 5, 9];
    assert(find(s, 4) == Some(2nat));
    assert(find(s, 7).is_none());
}

pub proof fn linear_search_verify() { example_linear(); search_finds_target(seq![1nat, 2, 3], 2, 0); }
pub fn main() { proof { linear_search_verify(); } }

} // verus!
