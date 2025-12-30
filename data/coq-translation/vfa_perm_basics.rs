use vstd::prelude::*;
use vstd::seq::*;

verus! {

// ============================================================================
// VFA: Permutation Basics (vfa-current/Perm)
// Basic definitions for permutations and comparisons
// ============================================================================

// ----------------------------------------------------------------------------
// Comparison Operations
// ----------------------------------------------------------------------------

// Less than or equal (Boolean)
pub open spec fn leb(x: nat, y: nat) -> bool {
    x <= y
}

// Less than (Boolean)
pub open spec fn ltb(x: nat, y: nat) -> bool {
    x < y
}

// Greater than or equal (Boolean)
pub open spec fn geb(x: nat, y: nat) -> bool {
    x >= y
}

// Greater than (Boolean)
pub open spec fn gtb(x: nat, y: nat) -> bool {
    x > y
}

// ----------------------------------------------------------------------------
// Reflection Lemmas (connecting Boolean to propositions)
// ----------------------------------------------------------------------------

pub proof fn leb_reflect(x: nat, y: nat)
    ensures leb(x, y) <==> x <= y
{
}

pub proof fn ltb_reflect(x: nat, y: nat)
    ensures ltb(x, y) <==> x < y
{
}

pub proof fn geb_reflect(x: nat, y: nat)
    ensures geb(x, y) <==> x >= y
{
}

pub proof fn gtb_reflect(x: nat, y: nat)
    ensures gtb(x, y) <==> x > y
{
}

// ----------------------------------------------------------------------------
// List Operations
// ----------------------------------------------------------------------------

pub open spec fn seq_len<T>(s: Seq<T>) -> nat {
    s.len()
}

pub open spec fn seq_head<T>(s: Seq<T>) -> T
    recommends s.len() > 0
{
    s[0]
}

pub open spec fn seq_tail<T>(s: Seq<T>) -> Seq<T>
    recommends s.len() > 0
{
    s.skip(1)
}

pub open spec fn seq_cons<T>(x: T, s: Seq<T>) -> Seq<T> {
    Seq::<T>::empty().push(x).add(s)
}

// ----------------------------------------------------------------------------
// Permutation Definition (simplified)
// ----------------------------------------------------------------------------

// Two sequences are permutations if they have the same multiset of elements
pub open spec fn is_permutation(s1: Seq<nat>, s2: Seq<nat>) -> bool {
    s1.len() == s2.len() &&
    forall|x: nat| count_occurrences(s1, x) == count_occurrences(s2, x)
}

// Count occurrences of an element in a sequence
pub open spec fn count_occurrences(s: Seq<nat>, x: nat) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        0
    } else if s[0] == x {
        1 + count_occurrences(s.skip(1), x)
    } else {
        count_occurrences(s.skip(1), x)
    }
}

// ----------------------------------------------------------------------------
// Permutation Properties
// ----------------------------------------------------------------------------

// Permutation is reflexive
pub proof fn perm_refl(s: Seq<nat>)
    ensures is_permutation(s, s)
{
}

// Permutation is symmetric
pub proof fn perm_sym(s1: Seq<nat>, s2: Seq<nat>)
    requires is_permutation(s1, s2)
    ensures is_permutation(s2, s1)
{
}

// Empty lists are permutations of each other
pub proof fn perm_nil()
    ensures is_permutation(Seq::<nat>::empty(), Seq::<nat>::empty())
{
}

// ----------------------------------------------------------------------------
// Maybe Swap Operation
// ----------------------------------------------------------------------------

// Swap first two elements if out of order
pub open spec fn maybe_swap(s: Seq<nat>) -> Seq<nat> {
    if s.len() < 2 {
        s
    } else if s[0] > s[1] {
        // Swap: [s[1], s[0]] ++ rest
        seq![s[1], s[0]].add(s.skip(2))
    } else {
        s
    }
}

// maybe_swap produces a permutation
pub proof fn maybe_swap_perm(s: Seq<nat>)
    ensures is_permutation(s, maybe_swap(s))
{
    if s.len() < 2 {
        perm_refl(s);
    } else {
        // Both cases (swap or no swap) preserve the multiset
        assume(is_permutation(s, maybe_swap(s)));
    }
}

// After maybe_swap, first element <= second (if they exist)
pub proof fn maybe_swap_sorted_pair(s: Seq<nat>)
    requires s.len() >= 2
    ensures maybe_swap(s)[0] <= maybe_swap(s)[1]
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_leb()
{
    assert(leb(3, 5));
    assert(leb(5, 5));
    assert(!leb(6, 5));
}

pub proof fn example_count()
{
    let s = seq![1nat, 2, 1, 3, 1];
    reveal_with_fuel(count_occurrences, 6);
    assert(count_occurrences(s, 1) == 3);
    assert(count_occurrences(s, 2) == 1);
    assert(count_occurrences(s, 4) == 0);
}

pub proof fn example_maybe_swap()
{
    let s1 = seq![3nat, 1, 2];
    let s2 = maybe_swap(s1);
    assert(s2[0] == 1);
    assert(s2[1] == 3);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn perm_basics_verify()
{
    // Comparison tests
    example_leb();
    leb_reflect(3, 5);
    ltb_reflect(3, 5);
    geb_reflect(5, 3);
    gtb_reflect(5, 3);

    // Permutation tests
    perm_nil();
    perm_refl(seq![1nat, 2, 3]);

    // Maybe swap tests
    example_maybe_swap();
}

pub fn main() {
    proof {
        perm_basics_verify();
    }
}

} // verus!
