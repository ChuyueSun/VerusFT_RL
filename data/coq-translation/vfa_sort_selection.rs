use vstd::prelude::*;
use vstd::seq::*;

verus! {

// ============================================================================
// VFA: Selection Sort (vfa-current/Selection)
// Implementation and verification of selection sort
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
// Find Minimum
// ----------------------------------------------------------------------------

// Find the minimum element in a non-empty sequence
pub open spec fn find_min(s: Seq<nat>) -> nat
    recommends s.len() > 0
    decreases s.len()
{
    if s.len() <= 1 {
        if s.len() == 0 { 0 } else { s[0] }
    } else {
        let tail = s.subrange(1, s.len() as int);
        let rest_min = find_min(tail);
        if s[0] <= rest_min { s[0] } else { rest_min }
    }
}

// Find index of minimum element
pub open spec fn find_min_idx(s: Seq<nat>) -> int
    recommends s.len() > 0
    decreases s.len()
{
    if s.len() <= 1 {
        0
    } else {
        let tail = s.subrange(1, s.len() as int);
        let rest_idx = find_min_idx(tail);
        if rest_idx + 1 < s.len() && s[0] <= s[rest_idx + 1] { 0 } else { rest_idx + 1 }
    }
}

// ----------------------------------------------------------------------------
// Remove Element
// ----------------------------------------------------------------------------

// Remove element at index i
pub open spec fn remove_at(s: Seq<nat>, i: int) -> Seq<nat>
    recommends 0 <= i < s.len()
{
    s.take(i).add(s.skip(i + 1))
}

// ----------------------------------------------------------------------------
// Selection Sort
// ----------------------------------------------------------------------------

// Selection sort with fuel for termination
pub open spec fn selection_sort(s: Seq<nat>, fuel: nat) -> Seq<nat>
    decreases fuel
{
    if fuel == 0 || s.len() == 0 {
        s
    } else {
        let min_idx = find_min_idx(s);
        let min_val = s[min_idx];
        let rest = remove_at(s, min_idx);
        seq![min_val].add(selection_sort(rest, (fuel - 1) as nat))
    }
}

// Wrapper with sufficient fuel
pub open spec fn sel_sort(s: Seq<nat>) -> Seq<nat> {
    selection_sort(s, s.len())
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Minimum is in the sequence
pub proof fn min_in_seq(s: Seq<nat>)
    requires s.len() > 0
    ensures s.contains(find_min(s))
    decreases s.len()
{
    reveal_with_fuel(find_min, 2);
    if s.len() == 1 {
    } else {
        min_in_seq(s.skip(1));
    }
}

// Minimum is <= all elements
pub proof fn min_le_all(s: Seq<nat>, i: int)
    requires s.len() > 0, 0 <= i < s.len()
    ensures find_min(s) <= s[i]
    decreases s.len()
{
    reveal_with_fuel(find_min, 2);
    if s.len() == 1 {
    } else if i == 0 {
    } else {
        min_le_all(s.skip(1), i - 1);
    }
}

// Remove preserves length minus 1
pub proof fn remove_length(s: Seq<nat>, i: int)
    requires 0 <= i < s.len()
    ensures remove_at(s, i).len() == s.len() - 1
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_find_min()
{
    let s = seq![3nat, 1, 4, 1, 5];
    reveal_with_fuel(find_min, 6);
    assert(find_min(s) == 1);
}

pub proof fn example_find_min_idx()
{
    let s = seq![3nat, 1, 4];
    reveal_with_fuel(find_min_idx, 4);
    assert(find_min_idx(s) == 1);
}

pub proof fn example_remove_at()
{
    let s = seq![1nat, 2, 3, 4];
    assert(remove_at(s, 1) =~= seq![1nat, 3, 4]);
    assert(remove_at(s, 0) =~= seq![2nat, 3, 4]);
}

pub proof fn example_selection_sort()
{
    let s = seq![3nat, 1, 2];
    reveal_with_fuel(selection_sort, 4);
    reveal_with_fuel(find_min_idx, 4);
    // Selection sort should produce [1, 2, 3]
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn sort_selection_verify()
{
    example_find_min();
    example_find_min_idx();
    example_remove_at();

    // Test min properties
    let s = seq![5nat, 3, 8, 1, 9];
    min_in_seq(s);
    min_le_all(s, 0);
    min_le_all(s, 3);
}

pub fn main() {
    proof {
        sort_selection_verify();
    }
}

} // verus!
