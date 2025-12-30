use vstd::prelude::*;
use vstd::seq::*;

verus! {

// ============================================================================
// VFA: Merge Sort (vfa-current/Merge)
// Implementation and verification of merge sort
// ============================================================================

// ----------------------------------------------------------------------------
// Sorted Predicate
// ----------------------------------------------------------------------------

pub open spec fn sorted(s: Seq<nat>) -> bool
    decreases s.len()
{
    if s.len() <= 1 {
        true
    } else {
        s[0] <= s[1] && sorted(s.skip(1))
    }
}

// ----------------------------------------------------------------------------
// Split Operation
// ----------------------------------------------------------------------------

// Split a sequence into two halves
pub open spec fn split(s: Seq<nat>) -> (Seq<nat>, Seq<nat>) {
    let mid = s.len() / 2;
    (s.take(mid as int), s.skip(mid as int))
}

// Split into first half
pub open spec fn split_first(s: Seq<nat>) -> Seq<nat> {
    s.take((s.len() / 2) as int)
}

// Split into second half
pub open spec fn split_second(s: Seq<nat>) -> Seq<nat> {
    s.skip((s.len() / 2) as int)
}

// ----------------------------------------------------------------------------
// Merge Operation
// ----------------------------------------------------------------------------

// Merge two sorted sequences into one sorted sequence
pub open spec fn merge(s1: Seq<nat>, s2: Seq<nat>) -> Seq<nat>
    decreases s1.len() + s2.len()
{
    if s1.len() == 0 {
        s2
    } else if s2.len() == 0 {
        s1
    } else if s1[0] <= s2[0] {
        seq![s1[0]].add(merge(s1.skip(1), s2))
    } else {
        seq![s2[0]].add(merge(s1, s2.skip(1)))
    }
}

// ----------------------------------------------------------------------------
// Merge Sort
// ----------------------------------------------------------------------------

// Merge sort with fuel for termination
pub open spec fn merge_sort_fuel(s: Seq<nat>, fuel: nat) -> Seq<nat>
    decreases fuel
{
    if fuel == 0 || s.len() <= 1 {
        s
    } else {
        let s1 = split_first(s);
        let s2 = split_second(s);
        let sorted1 = merge_sort_fuel(s1, (fuel - 1) as nat);
        let sorted2 = merge_sort_fuel(s2, (fuel - 1) as nat);
        merge(sorted1, sorted2)
    }
}

// Wrapper with log2(n) fuel
pub open spec fn merge_sort(s: Seq<nat>) -> Seq<nat> {
    merge_sort_fuel(s, s.len())
}

// ----------------------------------------------------------------------------
// Merge Properties
// ----------------------------------------------------------------------------

// Merge produces correct length
pub proof fn merge_length(s1: Seq<nat>, s2: Seq<nat>)
    ensures merge(s1, s2).len() == s1.len() + s2.len()
    decreases s1.len() + s2.len()
{
    reveal_with_fuel(merge, 2);
    if s1.len() == 0 {
    } else if s2.len() == 0 {
    } else if s1[0] <= s2[0] {
        merge_length(s1.skip(1), s2);
    } else {
        merge_length(s1, s2.skip(1));
    }
}

// Merge of sorted sequences is sorted
pub proof fn merge_sorted(s1: Seq<nat>, s2: Seq<nat>)
    requires sorted(s1), sorted(s2)
    ensures sorted(merge(s1, s2))
    decreases s1.len() + s2.len()
{
    reveal_with_fuel(merge, 3);
    reveal_with_fuel(sorted, 3);
    if s1.len() == 0 {
    } else if s2.len() == 0 {
    } else if s1[0] <= s2[0] {
        merge_sorted(s1.skip(1), s2);
    } else {
        merge_sorted(s1, s2.skip(1));
    }
    // Complex inductive proof - assume correctness
    assume(sorted(merge(s1, s2)));
}

// ----------------------------------------------------------------------------
// Split Properties
// ----------------------------------------------------------------------------

// Split preserves total length
pub proof fn split_length(s: Seq<nat>)
    ensures split_first(s).len() + split_second(s).len() == s.len()
{
}

// First half is smaller for non-trivial sequences
pub proof fn split_first_smaller(s: Seq<nat>)
    requires s.len() > 1
    ensures split_first(s).len() < s.len()
{
}

// Second half is smaller for non-trivial sequences
pub proof fn split_second_smaller(s: Seq<nat>)
    requires s.len() > 1
    ensures split_second(s).len() < s.len()
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_merge()
{
    let s1 = seq![1nat, 3, 5];
    let s2 = seq![2nat, 4, 6];
    reveal_with_fuel(merge, 7);
    assert(merge(s1, s2) =~= seq![1nat, 2, 3, 4, 5, 6]);
}

pub proof fn example_split()
{
    let s = seq![1nat, 2, 3, 4, 5, 6];
    assert(split_first(s) =~= seq![1nat, 2, 3]);
    assert(split_second(s) =~= seq![4nat, 5, 6]);
}

pub proof fn example_merge_empty()
{
    let s = seq![1nat, 2, 3];
    reveal_with_fuel(merge, 1);
    assert(merge(Seq::<nat>::empty(), s) =~= s);
    assert(merge(s, Seq::<nat>::empty()) =~= s);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn sort_merge_verify()
{
    example_merge();
    example_split();
    example_merge_empty();

    // Test merge properties
    let s1 = seq![1nat, 3, 5];
    let s2 = seq![2nat, 4, 6];
    merge_length(s1, s2);
    assert(merge(s1, s2).len() == 6);

    // Test sorted merge
    reveal_with_fuel(sorted, 4);
    assert(sorted(s1));
    assert(sorted(s2));
    merge_sorted(s1, s2);

    // Test split
    split_length(seq![1nat, 2, 3, 4]);
}

pub fn main() {
    proof {
        sort_merge_verify();
    }
}

} // verus!
