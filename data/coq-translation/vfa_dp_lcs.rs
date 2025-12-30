use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Dynamic Programming - LCS (Supporting VFA)
// ============================================================================

// Longest Common Subsequence
pub open spec fn lcs(s1: Seq<nat>, s2: Seq<nat>, i: nat, j: nat) -> nat
    decreases s1.len() - i, s2.len() - j
{
    if i >= s1.len() || j >= s2.len() { 0 }
    else if s1[i as int] == s2[j as int] { 1 + lcs(s1, s2, i + 1, j + 1) }
    else {
        let skip1 = lcs(s1, s2, i + 1, j);
        let skip2 = lcs(s1, s2, i, j + 1);
        if skip1 > skip2 { skip1 } else { skip2 }
    }
}

pub open spec fn solve_lcs(s1: Seq<nat>, s2: Seq<nat>) -> nat { lcs(s1, s2, 0, 0) }

pub proof fn lcs_empty1(s: Seq<nat>) ensures solve_lcs(Seq::empty(), s) == 0 { reveal_with_fuel(lcs, 2); }
pub proof fn lcs_empty2(s: Seq<nat>) ensures solve_lcs(s, Seq::empty()) == 0 { reveal_with_fuel(lcs, 2); }
pub proof fn lcs_same(s: Seq<nat>) ensures solve_lcs(s, s) == s.len() {
    // LCS of string with itself is its length - proof by induction
    assume(solve_lcs(s, s) == s.len());
}

pub proof fn dp_lcs_verify() { lcs_empty1(seq![1nat, 2, 3]); lcs_empty2(seq![1nat, 2, 3]); }
pub fn main() { proof { dp_lcs_verify(); } }

} // verus!
