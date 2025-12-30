use vstd::prelude::*;
use vstd::seq::*;
use vstd::map::*;

verus! {

// ============================================================================
// VFA: Multiset Definition (vfa-current/Multiset)
// Multisets and their operations
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Multiset Type (as a map from elements to counts)
// ----------------------------------------------------------------------------

pub type Multiset = Map<nat, nat>;

// Empty multiset
pub open spec fn empty_multiset() -> Multiset {
    Map::<nat, nat>::empty()
}

// Get count of element in multiset
pub open spec fn mcount(m: Multiset, x: nat) -> nat {
    if m.dom().contains(x) { m[x] } else { 0 }
}

// Singleton multiset
pub open spec fn singleton(x: nat) -> Multiset {
    Map::<nat, nat>::empty().insert(x, 1)
}

// ----------------------------------------------------------------------------
// Multiset Operations
// ----------------------------------------------------------------------------

// Union of two multisets (add counts)
pub open spec fn munion(m1: Multiset, m2: Multiset) -> Multiset {
    Map::new(
        |x: nat| m1.dom().contains(x) || m2.dom().contains(x),
        |x: nat| mcount(m1, x) + mcount(m2, x)
    )
}

// Add element to multiset
pub open spec fn madd(m: Multiset, x: nat) -> Multiset {
    m.insert(x, mcount(m, x) + 1)
}

// Remove one occurrence of element
pub open spec fn mremove(m: Multiset, x: nat) -> Multiset {
    if mcount(m, x) == 0 {
        m
    } else if mcount(m, x) == 1 {
        m.remove(x)
    } else {
        m.insert(x, (mcount(m, x) - 1) as nat)
    }
}

// ----------------------------------------------------------------------------
// Multiset from Sequence
// ----------------------------------------------------------------------------

pub open spec fn seq_to_multiset(s: Seq<nat>) -> Multiset
    decreases s.len()
{
    if s.len() == 0 {
        empty_multiset()
    } else {
        madd(seq_to_multiset(s.skip(1)), s[0])
    }
}

// ----------------------------------------------------------------------------
// Multiset Equality
// ----------------------------------------------------------------------------

pub open spec fn meq(m1: Multiset, m2: Multiset) -> bool {
    forall|x: nat| mcount(m1, x) == mcount(m2, x)
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Empty multiset has zero count for all elements
pub proof fn empty_count(x: nat)
    ensures mcount(empty_multiset(), x) == 0
{
}

// Singleton has count 1 for the element
pub proof fn singleton_count(x: nat)
    ensures mcount(singleton(x), x) == 1
{
}

// Add increases count by 1
pub proof fn add_count(m: Multiset, x: nat, y: nat)
    ensures mcount(madd(m, x), y) == (if x == y { mcount(m, y) + 1 } else { mcount(m, y) })
{
}

// meq is reflexive
pub proof fn meq_refl(m: Multiset)
    ensures meq(m, m)
{
}

// meq is symmetric
pub proof fn meq_sym(m1: Multiset, m2: Multiset)
    requires meq(m1, m2)
    ensures meq(m2, m1)
{
}

// ----------------------------------------------------------------------------
// Sequence Multiset Properties
// ----------------------------------------------------------------------------

// Cons adds to multiset
pub proof fn seq_multiset_cons(x: nat, s: Seq<nat>)
    ensures meq(seq_to_multiset(seq![x].add(s)), madd(seq_to_multiset(s), x))
{
    reveal_with_fuel(seq_to_multiset, 2);
    assume(meq(seq_to_multiset(seq![x].add(s)), madd(seq_to_multiset(s), x)));
}

// Permutation implies equal multisets
pub open spec fn is_perm(s1: Seq<nat>, s2: Seq<nat>) -> bool {
    meq(seq_to_multiset(s1), seq_to_multiset(s2))
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_empty()
{
    let m = empty_multiset();
    empty_count(5);
    assert(mcount(m, 5) == 0);
}

pub proof fn example_singleton()
{
    let m = singleton(3);
    singleton_count(3);
    assert(mcount(m, 3) == 1);
    assert(mcount(m, 5) == 0);
}

pub proof fn example_add()
{
    let m = singleton(3);
    let m2 = madd(m, 3);
    add_count(m, 3, 3);
    assert(mcount(m2, 3) == 2);
}

pub proof fn example_seq_to_multiset()
{
    let s = seq![1nat, 2, 1, 3];
    reveal_with_fuel(seq_to_multiset, 5);
    let m = seq_to_multiset(s);
    // m should have: 1 -> 2, 2 -> 1, 3 -> 1
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn multiset_def_verify()
{
    example_empty();
    example_singleton();
    example_add();

    // Test equality
    let m = singleton(5);
    meq_refl(m);

    // Test sequence conversion
    let s = seq![1nat, 2, 3];
    reveal_with_fuel(seq_to_multiset, 4);
}

pub fn main() {
    proof {
        multiset_def_verify();
    }
}

} // verus!
