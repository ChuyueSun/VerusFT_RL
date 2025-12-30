use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: BST Search (vfa-current/SearchTree)
// Search operations in BSTs
// ============================================================================

// ----------------------------------------------------------------------------
// Tree Type
// ----------------------------------------------------------------------------

pub enum Tree {
    E,
    T { left: Box<Tree>, key: nat, value: nat, right: Box<Tree> },
}

// ----------------------------------------------------------------------------
// Search Operations
// ----------------------------------------------------------------------------

// Lookup with default value
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

// Check if key exists
pub open spec fn contains(k: nat, t: Tree) -> bool
    decreases t
{
    match t {
        Tree::E => false,
        Tree::T { left, key, value: _, right } =>
            if k < key {
                contains(k, *left)
            } else if k > key {
                contains(k, *right)
            } else {
                true
            }
    }
}

// Find minimum key
pub open spec fn find_min(t: Tree) -> Option<nat>
    decreases t
{
    match t {
        Tree::E => None,
        Tree::T { left, key, value: _, right: _ } =>
            match *left {
                Tree::E => Some(key),
                _ => find_min(*left),
            }
    }
}

// Find maximum key
pub open spec fn find_max(t: Tree) -> Option<nat>
    decreases t
{
    match t {
        Tree::E => None,
        Tree::T { left: _, key, value: _, right } =>
            match *right {
                Tree::E => Some(key),
                _ => find_max(*right),
            }
    }
}

// ----------------------------------------------------------------------------
// BST Invariant
// ----------------------------------------------------------------------------

pub open spec fn forall_tree(t: Tree, p: spec_fn(nat) -> bool) -> bool
    decreases t
{
    match t {
        Tree::E => true,
        Tree::T { left, key, value: _, right } =>
            p(key) && forall_tree(*left, p) && forall_tree(*right, p),
    }
}

pub open spec fn all_lt(t: Tree, bound: nat) -> bool {
    forall_tree(t, |k: nat| k < bound)
}

pub open spec fn all_gt(t: Tree, bound: nat) -> bool {
    forall_tree(t, |k: nat| k > bound)
}

pub open spec fn is_bst(t: Tree) -> bool
    decreases t
{
    match t {
        Tree::E => true,
        Tree::T { left, key, value: _, right } =>
            all_lt(*left, key) &&
            all_gt(*right, key) &&
            is_bst(*left) &&
            is_bst(*right),
    }
}

// ----------------------------------------------------------------------------
// Search Properties
// ----------------------------------------------------------------------------

// Contains implies lookup doesn't return default (for BST)
pub proof fn contains_lookup(d: nat, k: nat, t: Tree)
    requires is_bst(t), contains(k, t)
    ensures lookup(d, k, t) == lookup(0, k, t)
    decreases t
{
    match t {
        Tree::E => {}
        Tree::T { left, key, value: _, right } => {
            if k < key {
                contains_lookup(d, k, *left);
            } else if k > key {
                contains_lookup(d, k, *right);
            }
        }
    }
}

// Not contains implies lookup returns default (for BST)
pub proof fn not_contains_lookup(d: nat, k: nat, t: Tree)
    requires is_bst(t), !contains(k, t)
    ensures lookup(d, k, t) == d
    decreases t
{
    match t {
        Tree::E => {}
        Tree::T { left, key, value: _, right } => {
            if k < key {
                not_contains_lookup(d, k, *left);
            } else if k > key {
                not_contains_lookup(d, k, *right);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_lookup()
{
    reveal_with_fuel(lookup, 4);
    let t = Tree::T {
        left: Box::new(Tree::T {
            left: Box::new(Tree::E),
            key: 3,
            value: 30,
            right: Box::new(Tree::E),
        }),
        key: 5,
        value: 50,
        right: Box::new(Tree::T {
            left: Box::new(Tree::E),
            key: 7,
            value: 70,
            right: Box::new(Tree::E),
        }),
    };

    assert(lookup(0, 5, t) == 50);
    assert(lookup(0, 3, t) == 30);
    assert(lookup(0, 7, t) == 70);
    assert(lookup(0, 1, t) == 0);
}

pub proof fn example_contains()
{
    reveal_with_fuel(contains, 3);
    let t = Tree::T {
        left: Box::new(Tree::E),
        key: 5,
        value: 50,
        right: Box::new(Tree::E),
    };

    assert(contains(5, t));
    assert(!contains(3, t));
}

pub proof fn example_find_min_max()
{
    reveal_with_fuel(find_min, 3);
    reveal_with_fuel(find_max, 3);
    let t = Tree::T {
        left: Box::new(Tree::T {
            left: Box::new(Tree::E),
            key: 1,
            value: 10,
            right: Box::new(Tree::E),
        }),
        key: 5,
        value: 50,
        right: Box::new(Tree::T {
            left: Box::new(Tree::E),
            key: 9,
            value: 90,
            right: Box::new(Tree::E),
        }),
    };

    assert(find_min(t) == Some(1nat));
    assert(find_max(t) == Some(9nat));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn bst_search_verify()
{
    example_lookup();
    example_contains();
    example_find_min_max();

    // Empty tree
    assert(!contains(5, Tree::E));
    assert(lookup(42, 5, Tree::E) == 42);
    assert(find_min(Tree::E).is_none());
    assert(find_max(Tree::E).is_none());
}

pub fn main() {
    proof {
        bst_search_verify();
    }
}

} // verus!
