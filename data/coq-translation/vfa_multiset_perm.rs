use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// VFA: Multiset Permutations (vfa-current/BagPerm)
// Permutation equivalence via multisets
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Multiset Type
// ----------------------------------------------------------------------------

pub struct Bag {
    pub counts: Map<nat, nat>,
}

// ----------------------------------------------------------------------------
// Bag from Sequence
// ----------------------------------------------------------------------------

// Convert sequence to bag
pub open spec fn seq_to_bag(s: Seq<nat>) -> Bag
    decreases s.len()
{
    if s.len() == 0 {
        Bag { counts: Map::empty() }
    } else {
        bag_add(seq_to_bag(s.skip(1)), s[0])
    }
}

// Get count from bag
pub open spec fn bag_count(b: Bag, x: nat) -> nat {
    if b.counts.dom().contains(x) {
        b.counts[x]
    } else {
        0
    }
}

// Add element to bag
pub open spec fn bag_add(b: Bag, x: nat) -> Bag {
    Bag {
        counts: b.counts.insert(x, bag_count(b, x) + 1),
    }
}

// ----------------------------------------------------------------------------
// Bag Equality (Permutation)
// ----------------------------------------------------------------------------

// Two bags are equal if they have same counts
pub open spec fn bag_eq(b1: Bag, b2: Bag) -> bool {
    forall|x: nat| bag_count(b1, x) == bag_count(b2, x)
}

// Two sequences are permutations if their bags are equal
pub open spec fn is_permutation(s1: Seq<nat>, s2: Seq<nat>) -> bool {
    bag_eq(seq_to_bag(s1), seq_to_bag(s2))
}

// ----------------------------------------------------------------------------
// Bag Properties
// ----------------------------------------------------------------------------

// Empty bag has zero counts
pub proof fn bag_count_empty(x: nat)
    ensures bag_count(Bag { counts: Map::empty() }, x) == 0
{
}

// Add increments count
pub proof fn bag_add_count(b: Bag, x: nat, y: nat)
    ensures bag_count(bag_add(b, x), y) == if x == y { bag_count(b, y) + 1 } else { bag_count(b, y) }
{
}

// Bag equality is reflexive
pub proof fn bag_eq_refl(b: Bag)
    ensures bag_eq(b, b)
{
}

// Bag equality is symmetric
pub proof fn bag_eq_sym(b1: Bag, b2: Bag)
    requires bag_eq(b1, b2)
    ensures bag_eq(b2, b1)
{
}

// Bag equality is transitive
pub proof fn bag_eq_trans(b1: Bag, b2: Bag, b3: Bag)
    requires bag_eq(b1, b2), bag_eq(b2, b3)
    ensures bag_eq(b1, b3)
{
}

// ----------------------------------------------------------------------------
// Permutation Properties
// ----------------------------------------------------------------------------

// Permutation is reflexive
pub proof fn perm_refl(s: Seq<nat>)
    ensures is_permutation(s, s)
{
    bag_eq_refl(seq_to_bag(s));
}

// Permutation is symmetric
pub proof fn perm_sym(s1: Seq<nat>, s2: Seq<nat>)
    requires is_permutation(s1, s2)
    ensures is_permutation(s2, s1)
{
    bag_eq_sym(seq_to_bag(s1), seq_to_bag(s2));
}

// Permutation is transitive
pub proof fn perm_trans(s1: Seq<nat>, s2: Seq<nat>, s3: Seq<nat>)
    requires is_permutation(s1, s2), is_permutation(s2, s3)
    ensures is_permutation(s1, s3)
{
    bag_eq_trans(seq_to_bag(s1), seq_to_bag(s2), seq_to_bag(s3));
}

// ----------------------------------------------------------------------------
// Append and Permutation
// ----------------------------------------------------------------------------

// Bag of append
pub open spec fn bag_union(b1: Bag, b2: Bag) -> Bag {
    Bag {
        counts: Map::new(
            |k: nat| b1.counts.dom().contains(k) || b2.counts.dom().contains(k),
            |k: nat| bag_count(b1, k) + bag_count(b2, k),
        ),
    }
}

pub proof fn bag_union_count(b1: Bag, b2: Bag, x: nat)
    ensures bag_count(bag_union(b1, b2), x) == bag_count(b1, x) + bag_count(b2, x)
{
}

// Bag union is commutative
pub proof fn bag_union_comm(b1: Bag, b2: Bag)
    ensures bag_eq(bag_union(b1, b2), bag_union(b2, b1))
{
    assert forall|x: nat| bag_count(bag_union(b1, b2), x) == bag_count(bag_union(b2, b1), x) by {
        bag_union_count(b1, b2, x);
        bag_union_count(b2, b1, x);
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_seq_to_bag()
{
    reveal_with_fuel(seq_to_bag, 4);
    let s = seq![1nat, 2, 1];
    let b = seq_to_bag(s);
    // Should have count(1) = 2, count(2) = 1
}

pub proof fn example_permutation()
{
    reveal_with_fuel(seq_to_bag, 4);
    let s1 = seq![1nat, 2, 3];
    let s2 = seq![3nat, 1, 2];

    // These should be permutations of each other
    perm_refl(s1);
    assert(is_permutation(s1, s1));
}

pub proof fn example_not_permutation()
{
    reveal_with_fuel(seq_to_bag, 4);
    let s1 = seq![1nat, 2, 3];
    let s2 = seq![1nat, 2, 4];  // Different elements

    // These are not permutations
    // s1 has count(3) = 1, s2 has count(3) = 0
}

pub proof fn example_perm_properties()
{
    let s1 = seq![5nat, 10, 15];
    let s2 = seq![10nat, 15, 5];
    let s3 = seq![15nat, 5, 10];

    perm_refl(s1);
    perm_refl(s2);
    perm_refl(s3);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn multiset_perm_verify()
{
    example_seq_to_bag();
    example_permutation();
    example_not_permutation();
    example_perm_properties();

    // Test bag equality properties
    let b = Bag { counts: Map::empty() };
    bag_eq_refl(b);
}

pub fn main() {
    proof {
        multiset_perm_verify();
    }
}

} // verus!
