use vstd::prelude::*;
use vstd::set::*;

verus! {

// ============================================================================
// VFA: Abstract Data Type - Set (vfa-current/ADT)
// Set ADT with algebraic specification
// ============================================================================

broadcast use vstd::set::group_set_axioms;

// ----------------------------------------------------------------------------
// Set Operations (Using VSTd Set)
// ----------------------------------------------------------------------------

// Empty set
pub open spec fn set_empty() -> Set<nat> {
    Set::empty()
}

// Singleton set
pub open spec fn set_singleton(x: nat) -> Set<nat> {
    Set::empty().insert(x)
}

// Insert element
pub open spec fn set_insert(x: nat, s: Set<nat>) -> Set<nat> {
    s.insert(x)
}

// Remove element
pub open spec fn set_remove(x: nat, s: Set<nat>) -> Set<nat> {
    s.remove(x)
}

// Check membership
pub open spec fn set_contains(x: nat, s: Set<nat>) -> bool {
    s.contains(x)
}

// Union
pub open spec fn set_union(s1: Set<nat>, s2: Set<nat>) -> Set<nat> {
    s1.union(s2)
}

// Intersection
pub open spec fn set_intersect(s1: Set<nat>, s2: Set<nat>) -> Set<nat> {
    s1.intersect(s2)
}

// Difference
pub open spec fn set_difference(s1: Set<nat>, s2: Set<nat>) -> Set<nat> {
    s1.difference(s2)
}

// Subset
pub open spec fn set_subset(s1: Set<nat>, s2: Set<nat>) -> bool {
    s1.subset_of(s2)
}

// ----------------------------------------------------------------------------
// Set Properties
// ----------------------------------------------------------------------------

// Empty set has no elements
pub proof fn empty_no_elements(x: nat)
    ensures !set_contains(x, set_empty())
{
}

// Insert makes element present
pub proof fn insert_contains(x: nat, s: Set<nat>)
    ensures set_contains(x, set_insert(x, s))
{
}

// Insert preserves other elements
pub proof fn insert_preserves(x: nat, y: nat, s: Set<nat>)
    requires x != y
    ensures set_contains(y, set_insert(x, s)) == set_contains(y, s)
{
}

// Remove makes element absent
pub proof fn remove_not_contains(x: nat, s: Set<nat>)
    ensures !set_contains(x, set_remove(x, s))
{
}

// Remove preserves other elements
pub proof fn remove_preserves(x: nat, y: nat, s: Set<nat>)
    requires x != y
    ensures set_contains(y, set_remove(x, s)) == set_contains(y, s)
{
}

// ----------------------------------------------------------------------------
// Union Properties
// ----------------------------------------------------------------------------

pub proof fn union_contains(x: nat, s1: Set<nat>, s2: Set<nat>)
    ensures set_contains(x, set_union(s1, s2)) ==
            (set_contains(x, s1) || set_contains(x, s2))
{
}

pub proof fn union_comm(s1: Set<nat>, s2: Set<nat>)
    ensures set_union(s1, s2) == set_union(s2, s1)
{
}

pub proof fn union_assoc(s1: Set<nat>, s2: Set<nat>, s3: Set<nat>)
    ensures set_union(set_union(s1, s2), s3) == set_union(s1, set_union(s2, s3))
{
    assert(set_union(set_union(s1, s2), s3) =~= set_union(s1, set_union(s2, s3)));
}

pub proof fn union_empty(s: Set<nat>)
    ensures set_union(s, set_empty()) == s
{
    assert(set_union(s, set_empty()) =~= s);
}

// ----------------------------------------------------------------------------
// Intersection Properties
// ----------------------------------------------------------------------------

pub proof fn intersect_contains(x: nat, s1: Set<nat>, s2: Set<nat>)
    ensures set_contains(x, set_intersect(s1, s2)) ==
            (set_contains(x, s1) && set_contains(x, s2))
{
}

pub proof fn intersect_comm(s1: Set<nat>, s2: Set<nat>)
    ensures set_intersect(s1, s2) == set_intersect(s2, s1)
{
}

pub proof fn intersect_empty(s: Set<nat>)
    ensures set_intersect(s, set_empty()) == set_empty()
{
    assert(set_intersect(s, set_empty()) =~= set_empty());
}

// ----------------------------------------------------------------------------
// Difference Properties
// ----------------------------------------------------------------------------

pub proof fn difference_contains(x: nat, s1: Set<nat>, s2: Set<nat>)
    ensures set_contains(x, set_difference(s1, s2)) ==
            (set_contains(x, s1) && !set_contains(x, s2))
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_basic()
{
    let s = set_empty();
    empty_no_elements(5);
    assert(!set_contains(5, s));

    let s1 = set_insert(5, s);
    insert_contains(5, s);
    assert(set_contains(5, s1));

    let s2 = set_remove(5, s1);
    remove_not_contains(5, s1);
    assert(!set_contains(5, s2));
}

pub proof fn example_union()
{
    let s1 = set_insert(1, set_insert(2, set_empty()));
    let s2 = set_insert(2, set_insert(3, set_empty()));
    let u = set_union(s1, s2);

    union_contains(1, s1, s2);
    union_contains(2, s1, s2);
    union_contains(3, s1, s2);

    assert(set_contains(1, u));
    assert(set_contains(2, u));
    assert(set_contains(3, u));
}

pub proof fn example_intersect()
{
    let s1 = set_insert(1, set_insert(2, set_empty()));
    let s2 = set_insert(2, set_insert(3, set_empty()));
    let i = set_intersect(s1, s2);

    intersect_contains(1, s1, s2);
    intersect_contains(2, s1, s2);
    intersect_contains(3, s1, s2);

    assert(!set_contains(1, i));
    assert(set_contains(2, i));  // Common element
    assert(!set_contains(3, i));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn adt_set_verify()
{
    example_basic();
    example_union();
    example_intersect();

    // Test properties
    union_comm(set_singleton(1), set_singleton(2));
    intersect_comm(set_singleton(1), set_singleton(2));
}

pub fn main() {
    proof {
        adt_set_verify();
    }
}

} // verus!
