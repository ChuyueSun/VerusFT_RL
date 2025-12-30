use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Binary Search Tree Definition (vfa-current/SearchTree)
// BST structure and basic operations
// ============================================================================

// ----------------------------------------------------------------------------
// Tree Type
// ----------------------------------------------------------------------------

pub enum Tree {
    E,  // Empty
    T { left: Box<Tree>, key: nat, value: nat, right: Box<Tree> },
}

// ----------------------------------------------------------------------------
// Basic Operations
// ----------------------------------------------------------------------------

// Empty tree
pub open spec fn empty_tree() -> Tree {
    Tree::E
}

// Lookup with default
pub open spec fn lookup(d: nat, k: nat, t: Tree) -> nat
    decreases t
{
    match t {
        Tree::E => d,
        Tree::T { left, key, value, right } =>
            if k < key {
                lookup(d, k, *left)
            } else if k > key {
                lookup(d, k, *right)
            } else {
                value
            }
    }
}

// Insert key-value pair
pub open spec fn insert(k: nat, v: nat, t: Tree) -> Tree
    decreases t
{
    match t {
        Tree::E => Tree::T {
            left: Box::new(Tree::E),
            key: k,
            value: v,
            right: Box::new(Tree::E),
        },
        Tree::T { left, key, value, right } =>
            if k < key {
                Tree::T {
                    left: Box::new(insert(k, v, *left)),
                    key,
                    value,
                    right,
                }
            } else if k > key {
                Tree::T {
                    left,
                    key,
                    value,
                    right: Box::new(insert(k, v, *right)),
                }
            } else {
                Tree::T { left, key: k, value: v, right }
            }
    }
}

// Check if key is bound
pub open spec fn bound(k: nat, t: Tree) -> bool
    decreases t
{
    match t {
        Tree::E => false,
        Tree::T { left, key, value: _, right } =>
            if k < key {
                bound(k, *left)
            } else if k > key {
                bound(k, *right)
            } else {
                true
            }
    }
}

// ----------------------------------------------------------------------------
// Tree Size
// ----------------------------------------------------------------------------

pub open spec fn tree_size(t: Tree) -> nat
    decreases t
{
    match t {
        Tree::E => 0,
        Tree::T { left, key: _, value: _, right } =>
            1 + tree_size(*left) + tree_size(*right),
    }
}

// ----------------------------------------------------------------------------
// Algebraic Specifications
// ----------------------------------------------------------------------------

// Lookup in empty tree returns default
pub proof fn lookup_empty(d: nat, k: nat)
    ensures lookup(d, k, empty_tree()) == d
{
}

// Lookup after insert (same key)
pub proof fn lookup_insert_eq(d: nat, k: nat, v: nat, t: Tree)
    ensures lookup(d, k, insert(k, v, t)) == v
    decreases t
{
    match t {
        Tree::E => {}
        Tree::T { left, key, value: _, right } => {
            if k < key {
                lookup_insert_eq(d, k, v, *left);
            } else if k > key {
                lookup_insert_eq(d, k, v, *right);
            } else {
            }
        }
    }
}

// Lookup after insert (different key)
pub proof fn lookup_insert_neq(d: nat, k1: nat, k2: nat, v: nat, t: Tree)
    requires k1 != k2
    ensures lookup(d, k1, insert(k2, v, t)) == lookup(d, k1, t)
    decreases t
{
    match t {
        Tree::E => {}
        Tree::T { left, key, value: _, right } => {
            if k2 < key {
                if k1 < key {
                    lookup_insert_neq(d, k1, k2, v, *left);
                }
            } else if k2 > key {
                if k1 > key {
                    lookup_insert_neq(d, k1, k2, v, *right);
                }
            }
        }
    }
}

// Bound after insert (same key)
pub proof fn bound_insert_eq(k: nat, v: nat, t: Tree)
    ensures bound(k, insert(k, v, t))
    decreases t
{
    match t {
        Tree::E => {}
        Tree::T { left, key, value: _, right } => {
            if k < key {
                bound_insert_eq(k, v, *left);
            } else if k > key {
                bound_insert_eq(k, v, *right);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_empty()
{
    lookup_empty(0, 5);
    assert(lookup(0, 5, empty_tree()) == 0);
    assert(!bound(5, empty_tree()));
}

pub proof fn example_insert_lookup()
{
    let t = insert(5, 10, empty_tree());
    lookup_insert_eq(0, 5, 10, empty_tree());
    assert(lookup(0, 5, t) == 10);

    lookup_insert_neq(0, 3, 5, 10, empty_tree());
    assert(lookup(0, 3, t) == 0);
}

pub proof fn example_bound()
{
    let t = insert(5, 10, empty_tree());
    bound_insert_eq(5, 10, empty_tree());
    reveal_with_fuel(bound, 2);
    reveal_with_fuel(insert, 2);
    assert(bound(5, t));
    assert(!bound(3, t));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn bst_def_verify()
{
    example_empty();
    example_insert_lookup();
    example_bound();

    // Chain of inserts
    let t1 = insert(5, 50, empty_tree());
    let t2 = insert(3, 30, t1);
    let t3 = insert(7, 70, t2);

    lookup_insert_eq(0, 5, 50, empty_tree());
    lookup_insert_eq(0, 3, 30, t1);
    lookup_insert_eq(0, 7, 70, t2);
}

pub fn main() {
    proof {
        bst_def_verify();
    }
}

} // verus!
