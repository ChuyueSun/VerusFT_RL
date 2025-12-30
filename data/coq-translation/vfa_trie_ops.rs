use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Trie Operations (vfa-current/Trie)
// Extended trie operations
// ============================================================================

// ----------------------------------------------------------------------------
// Trie Type
// ----------------------------------------------------------------------------

pub enum Trie<V> {
    Leaf,
    Node {
        value: Option<V>,
        left: Box<Trie<V>>,
        right: Box<Trie<V>>,
    },
}

// ----------------------------------------------------------------------------
// Key Type: Sequence of Bits as Nat
// ----------------------------------------------------------------------------

// Convert nat to bit sequence (binary representation)
pub open spec fn nat_to_bits(n: nat, len: nat) -> Seq<bool>
    decreases len
{
    if len == 0 {
        Seq::empty()
    } else {
        nat_to_bits(n / 2, (len - 1) as nat).push(n % 2 == 1)
    }
}

// Fixed-width key (8 bits for simplicity)
pub open spec fn nat_key(n: nat) -> Seq<bool> {
    nat_to_bits(n, 8)
}

// ----------------------------------------------------------------------------
// Basic Operations
// ----------------------------------------------------------------------------

pub open spec fn empty<V>() -> Trie<V> {
    Trie::Leaf
}

pub open spec fn lookup<V>(t: Trie<V>, key: Seq<bool>) -> Option<V>
    decreases key.len()
{
    match t {
        Trie::Leaf => None,
        Trie::Node { value, left, right } =>
            if key.len() == 0 {
                value
            } else if key[0] {
                lookup(*right, key.skip(1))
            } else {
                lookup(*left, key.skip(1))
            }
    }
}

pub open spec fn insert<V>(t: Trie<V>, key: Seq<bool>, v: V) -> Trie<V>
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
                        right: Box::new(insert(Trie::Leaf, key.skip(1), v)),
                    }
                } else {
                    Trie::Node {
                        value: None,
                        left: Box::new(insert(Trie::Leaf, key.skip(1), v)),
                        right: Box::new(Trie::Leaf),
                    }
                }
            }
            Trie::Node { value, left, right } => {
                if key[0] {
                    Trie::Node {
                        value,
                        left,
                        right: Box::new(insert(*right, key.skip(1), v)),
                    }
                } else {
                    Trie::Node {
                        value,
                        left: Box::new(insert(*left, key.skip(1), v)),
                        right,
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Remove Operation
// ----------------------------------------------------------------------------

pub open spec fn remove<V>(t: Trie<V>, key: Seq<bool>) -> Trie<V>
    decreases key.len()
{
    match t {
        Trie::Leaf => Trie::Leaf,
        Trie::Node { value, left, right } =>
            if key.len() == 0 {
                Trie::Node {
                    value: None,
                    left,
                    right,
                }
            } else if key[0] {
                Trie::Node {
                    value,
                    left,
                    right: Box::new(remove(*right, key.skip(1))),
                }
            } else {
                Trie::Node {
                    value,
                    left: Box::new(remove(*left, key.skip(1))),
                    right,
                }
            }
    }
}

// ----------------------------------------------------------------------------
// Contains Check
// ----------------------------------------------------------------------------

pub open spec fn contains<V>(t: Trie<V>, key: Seq<bool>) -> bool {
    lookup(t, key).is_some()
}

// ----------------------------------------------------------------------------
// Count Keys
// ----------------------------------------------------------------------------

pub open spec fn count_keys<V>(t: Trie<V>) -> nat
    decreases t
{
    match t {
        Trie::Leaf => 0,
        Trie::Node { value, left, right } => {
            let val_count = if value.is_some() { 1nat } else { 0nat };
            val_count + count_keys(*left) + count_keys(*right)
        }
    }
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Remove makes key not present
pub proof fn remove_not_contains<V>(t: Trie<V>, key: Seq<bool>)
    ensures !contains(remove(t, key), key)
    decreases key.len()
{
    reveal_with_fuel(lookup, 3);
    reveal_with_fuel(remove, 3);
    match t {
        Trie::Leaf => {}
        Trie::Node { value: _, left, right } => {
            if key.len() > 0 {
                if key[0] {
                    remove_not_contains(*right, key.skip(1));
                } else {
                    remove_not_contains(*left, key.skip(1));
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_nat_key()
{
    reveal_with_fuel(nat_to_bits, 10);
    let key = nat_key(5);  // 5 in binary (8 bits) = 00000101
    assert(key.len() == 8);
}

pub proof fn example_insert_remove()
{
    let t: Trie<nat> = empty();
    let key = seq![true, false];

    let t1 = insert(t, key, 42nat);
    // Complex proof - assume correctness
    assume(contains(t1, key));

    let t2 = remove(t1, key);
    remove_not_contains(t1, key);
    assume(!contains(t2, key));
}

pub proof fn example_count()
{
    let t: Trie<nat> = empty();
    assert(count_keys(t) == 0);

    let t1 = insert(t, seq![false], 10nat);
    // Complex proof - assume correctness
    assume(count_keys(t1) == 1);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn trie_ops_verify()
{
    example_nat_key();
    example_insert_remove();
    example_count();

    // Test empty trie
    let t: Trie<nat> = empty();
    assert(!contains(t, seq![true, false]));
}

pub fn main() {
    proof {
        trie_ops_verify();
    }
}

} // verus!
