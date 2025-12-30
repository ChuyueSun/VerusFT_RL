use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// VFA: Trie Definition (vfa-current/Trie)
// Trie data structure for string maps
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Trie Type (Binary Trie for simplicity)
// ----------------------------------------------------------------------------

// A trie where keys are sequences of bits (booleans)
pub enum Trie<V> {
    Leaf,
    Node {
        value: Option<V>,
        left: Box<Trie<V>>,   // false branch
        right: Box<Trie<V>>,  // true branch
    },
}

// ----------------------------------------------------------------------------
// Basic Operations
// ----------------------------------------------------------------------------

// Empty trie
pub open spec fn trie_empty<V>() -> Trie<V> {
    Trie::Leaf
}

// Lookup in trie
pub open spec fn trie_lookup<V>(t: Trie<V>, key: Seq<bool>) -> Option<V>
    decreases key.len()
{
    match t {
        Trie::Leaf => None,
        Trie::Node { value, left, right } =>
            if key.len() == 0 {
                value
            } else if key[0] {
                trie_lookup(*right, key.skip(1))
            } else {
                trie_lookup(*left, key.skip(1))
            }
    }
}

// Insert into trie
pub open spec fn trie_insert<V>(t: Trie<V>, key: Seq<bool>, v: V) -> Trie<V>
    decreases key.len()
{
    if key.len() == 0 {
        match t {
            Trie::Leaf => Trie::Node {
                value: Some(v),
                left: Box::new(Trie::Leaf),
                right: Box::new(Trie::Leaf),
            },
            Trie::Node { value: _, left, right } => Trie::Node {
                value: Some(v),
                left,
                right,
            },
        }
    } else {
        match t {
            Trie::Leaf => {
                if key[0] {
                    Trie::Node {
                        value: None,
                        left: Box::new(Trie::Leaf),
                        right: Box::new(trie_insert(Trie::Leaf, key.skip(1), v)),
                    }
                } else {
                    Trie::Node {
                        value: None,
                        left: Box::new(trie_insert(Trie::Leaf, key.skip(1), v)),
                        right: Box::new(Trie::Leaf),
                    }
                }
            }
            Trie::Node { value, left, right } => {
                if key[0] {
                    Trie::Node {
                        value,
                        left,
                        right: Box::new(trie_insert(*right, key.skip(1), v)),
                    }
                } else {
                    Trie::Node {
                        value,
                        left: Box::new(trie_insert(*left, key.skip(1), v)),
                        right,
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Lookup in empty trie returns None
pub proof fn trie_lookup_empty<V>(key: Seq<bool>)
    ensures trie_lookup::<V>(trie_empty(), key).is_none()
{
}

// Lookup after insert (same key)
pub proof fn trie_lookup_insert_eq<V>(t: Trie<V>, key: Seq<bool>, v: V)
    ensures trie_lookup(trie_insert(t, key, v), key) == Some(v)
    decreases key.len()
{
    reveal_with_fuel(trie_lookup, 3);
    reveal_with_fuel(trie_insert, 3);
    if key.len() == 0 {
        // Base case: empty key
    } else {
        match t {
            Trie::Leaf => {
                if key[0] {
                    trie_lookup_insert_eq(Trie::Leaf, key.skip(1), v);
                } else {
                    trie_lookup_insert_eq(Trie::Leaf, key.skip(1), v);
                }
            }
            Trie::Node { value: _, left, right } => {
                if key[0] {
                    trie_lookup_insert_eq(*right, key.skip(1), v);
                } else {
                    trie_lookup_insert_eq(*left, key.skip(1), v);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Trie Size
// ----------------------------------------------------------------------------

pub open spec fn trie_size<V>(t: Trie<V>) -> nat
    decreases t
{
    match t {
        Trie::Leaf => 0,
        Trie::Node { value, left, right } => {
            let val_count = if value.is_some() { 1nat } else { 0nat };
            val_count + trie_size(*left) + trie_size(*right)
        }
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_empty_trie()
{
    let t: Trie<nat> = trie_empty();
    trie_lookup_empty::<nat>(seq![true, false]);
    assert(trie_lookup(t, seq![true, false]).is_none());
}

pub proof fn example_insert_lookup()
{
    let t: Trie<nat> = trie_empty();
    let key = seq![true, false, true];
    let t1 = trie_insert(t, key, 42nat);

    trie_lookup_insert_eq(t, key, 42nat);
    assert(trie_lookup(t1, key) == Some(42nat));
}

pub proof fn example_multiple_inserts()
{
    let t: Trie<nat> = trie_empty();
    let k1 = seq![false];
    let k2 = seq![true];

    let t1 = trie_insert(t, k1, 10nat);
    trie_lookup_insert_eq(t, k1, 10nat);

    let t2 = trie_insert(t1, k2, 20nat);
    trie_lookup_insert_eq(t1, k2, 20nat);
    assert(trie_lookup(t2, k2) == Some(20nat));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn trie_def_verify()
{
    example_empty_trie();
    example_insert_lookup();
    example_multiple_inserts();

    // Test trie size
    let t: Trie<nat> = trie_empty();
    assert(trie_size(t) == 0);
}

pub fn main() {
    proof {
        trie_def_verify();
    }
}

} // verus!
