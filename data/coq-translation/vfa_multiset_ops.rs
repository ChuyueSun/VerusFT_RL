use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// VFA: Multiset Operations (vfa-current/Multiset)
// Extended multiset operations and properties
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Multiset Type (as Map from elements to counts)
// ----------------------------------------------------------------------------

pub struct Multiset {
    pub counts: Map<nat, nat>,
}

// ----------------------------------------------------------------------------
// Basic Operations
// ----------------------------------------------------------------------------

// Empty multiset
pub open spec fn empty() -> Multiset {
    Multiset { counts: Map::empty() }
}

// Singleton multiset
pub open spec fn singleton(x: nat) -> Multiset {
    Multiset { counts: Map::empty().insert(x, 1) }
}

// Get count of element
pub open spec fn count(m: Multiset, x: nat) -> nat {
    if m.counts.dom().contains(x) {
        m.counts[x]
    } else {
        0
    }
}

// Add one occurrence
pub open spec fn add(m: Multiset, x: nat) -> Multiset {
    Multiset {
        counts: m.counts.insert(x, count(m, x) + 1),
    }
}

// Remove one occurrence
pub open spec fn remove(m: Multiset, x: nat) -> Multiset {
    if count(m, x) == 0 {
        m
    } else {
        Multiset {
            counts: m.counts.insert(x, (count(m, x) - 1) as nat),
        }
    }
}

// Union (add counts)
pub open spec fn union(m1: Multiset, m2: Multiset) -> Multiset {
    Multiset {
        counts: Map::new(
            |k: nat| m1.counts.dom().contains(k) || m2.counts.dom().contains(k),
            |k: nat| count(m1, k) + count(m2, k),
        ),
    }
}

// Intersection (min counts)
pub open spec fn intersection(m1: Multiset, m2: Multiset) -> Multiset {
    Multiset {
        counts: Map::new(
            |k: nat| m1.counts.dom().contains(k) && m2.counts.dom().contains(k),
            |k: nat| if count(m1, k) <= count(m2, k) { count(m1, k) } else { count(m2, k) },
        ),
    }
}

// Difference (subtract counts, min 0)
pub open spec fn difference(m1: Multiset, m2: Multiset) -> Multiset {
    Multiset {
        counts: Map::new(
            |k: nat| m1.counts.dom().contains(k),
            |k: nat| if count(m1, k) >= count(m2, k) { (count(m1, k) - count(m2, k)) as nat } else { 0nat },
        ),
    }
}

// ----------------------------------------------------------------------------
// Predicates
// ----------------------------------------------------------------------------

// Check if multiset is empty
pub open spec fn is_empty(m: Multiset) -> bool {
    forall|x: nat| count(m, x) == 0
}

// Check membership
pub open spec fn contains(m: Multiset, x: nat) -> bool {
    count(m, x) > 0
}

// Submultiset relation
pub open spec fn submultiset(m1: Multiset, m2: Multiset) -> bool {
    forall|x: nat| count(m1, x) <= count(m2, x)
}

// Multiset equality
pub open spec fn multiset_eq(m1: Multiset, m2: Multiset) -> bool {
    forall|x: nat| count(m1, x) == count(m2, x)
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Empty multiset has zero count
pub proof fn count_empty(x: nat)
    ensures count(empty(), x) == 0
{
}

// Singleton has count 1 for element, 0 otherwise
pub proof fn count_singleton(x: nat, y: nat)
    ensures count(singleton(x), y) == if x == y { 1nat } else { 0nat }
{
}

// Add increments count
pub proof fn count_add(m: Multiset, x: nat, y: nat)
    ensures count(add(m, x), y) == if x == y { count(m, y) + 1 } else { count(m, y) }
{
}

// Union sums counts
pub proof fn count_union(m1: Multiset, m2: Multiset, x: nat)
    ensures count(union(m1, m2), x) == count(m1, x) + count(m2, x)
{
}

// ----------------------------------------------------------------------------
// Union Properties
// ----------------------------------------------------------------------------

// Union is commutative
pub proof fn union_comm(m1: Multiset, m2: Multiset)
    ensures multiset_eq(union(m1, m2), union(m2, m1))
{
    assert forall|x: nat| count(union(m1, m2), x) == count(union(m2, m1), x) by {
        count_union(m1, m2, x);
        count_union(m2, m1, x);
    }
}

// Union is associative
pub proof fn union_assoc(m1: Multiset, m2: Multiset, m3: Multiset)
    ensures multiset_eq(union(union(m1, m2), m3), union(m1, union(m2, m3)))
{
    assert forall|x: nat| count(union(union(m1, m2), m3), x) ==
        count(union(m1, union(m2, m3)), x) by {
        count_union(m1, m2, x);
        count_union(m2, m3, x);
        count_union(union(m1, m2), m3, x);
        count_union(m1, union(m2, m3), x);
    }
}

// Empty is identity for union
pub proof fn union_empty_left(m: Multiset)
    ensures multiset_eq(union(empty(), m), m)
{
    assert forall|x: nat| count(union(empty(), m), x) == count(m, x) by {
        count_empty(x);
        count_union(empty(), m, x);
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_basic()
{
    let m = empty();
    count_empty(5);
    assert(count(m, 5) == 0);

    let m1 = add(m, 5);
    count_add(m, 5, 5);
    assert(count(m1, 5) == 1);

    let m2 = add(m1, 5);
    count_add(m1, 5, 5);
    assert(count(m2, 5) == 2);
}

pub proof fn example_singleton()
{
    let m = singleton(42);
    count_singleton(42, 42);
    assert(count(m, 42) == 1);

    count_singleton(42, 10);
    assert(count(m, 10) == 0);
}

pub proof fn example_union()
{
    let m1 = add(add(empty(), 1), 2);
    let m2 = add(add(empty(), 2), 3);
    let u = union(m1, m2);

    count_union(m1, m2, 2);
    // m1 has 1 copy of 2, m2 has 1 copy of 2
    // union should have 2 copies of 2
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn multiset_ops_verify()
{
    example_basic();
    example_singleton();
    example_union();

    // Test properties
    let m1 = singleton(10);
    let m2 = singleton(20);
    union_comm(m1, m2);
    union_empty_left(m1);
}

pub fn main() {
    proof {
        multiset_ops_verify();
    }
}

} // verus!
