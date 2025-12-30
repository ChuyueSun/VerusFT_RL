use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Sorting Properties (Supporting VFA)
// General properties of sorting algorithms
// ============================================================================

pub open spec fn sorted(s: Seq<nat>) -> bool decreases s.len() {
    s.len() <= 1 || (s[0] <= s[1] && sorted(s.skip(1)))
}

pub open spec fn is_permutation(s1: Seq<nat>, s2: Seq<nat>) -> bool {
    s1.len() == s2.len() && forall|x: nat| count(s1, x) == count(s2, x)
}

pub open spec fn count(s: Seq<nat>, x: nat) -> nat decreases s.len() {
    if s.len() == 0 { 0 }
    else if s[0] == x { 1 + count(s.skip(1), x) }
    else { count(s.skip(1), x) }
}

pub open spec fn is_sort_of(sorted_s: Seq<nat>, original: Seq<nat>) -> bool {
    sorted(sorted_s) && is_permutation(sorted_s, original)
}

pub proof fn sorted_empty() ensures sorted(Seq::<nat>::empty()) {}
pub proof fn sorted_singleton(x: nat) ensures sorted(seq![x]) { reveal_with_fuel(sorted, 2); }

pub proof fn perm_refl(s: Seq<nat>) ensures is_permutation(s, s) {}
pub proof fn perm_sym(s1: Seq<nat>, s2: Seq<nat>)
    requires is_permutation(s1, s2)
    ensures is_permutation(s2, s1)
{}

pub proof fn example_sort_props() {
    sorted_empty();
    sorted_singleton(5);
    perm_refl(seq![1nat, 2, 3]);
}

pub proof fn sort_props_verify() { example_sort_props(); }
pub fn main() { proof { sort_props_verify(); } }

} // verus!
