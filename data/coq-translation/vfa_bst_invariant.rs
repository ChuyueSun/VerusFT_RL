use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: BST Invariant (vfa-current/SearchTree)
// BST ordering invariant and its preservation
// ============================================================================

// ----------------------------------------------------------------------------
// Tree Type
// ----------------------------------------------------------------------------

pub enum Tree {
    E,
    T { left: Box<Tree>, key: nat, value: nat, right: Box<Tree> },
}

// ----------------------------------------------------------------------------
// ForallT - Predicate holds at all nodes
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

// All keys less than bound
pub open spec fn all_lt(t: Tree, bound: nat) -> bool {
    forall_tree(t, |k: nat| k < bound)
}

// All keys greater than bound
pub open spec fn all_gt(t: Tree, bound: nat) -> bool {
    forall_tree(t, |k: nat| k > bound)
}

// ----------------------------------------------------------------------------
// BST Invariant
// ----------------------------------------------------------------------------

// A tree is a valid BST if:
// - For each node, all left keys < node key < all right keys
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
// Operations
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Insert Preserves BST
// ----------------------------------------------------------------------------

// Helper: insert preserves all_lt
pub proof fn insert_all_lt(k: nat, v: nat, t: Tree, bound: nat)
    requires all_lt(t, bound), k < bound
    ensures all_lt(insert(k, v, t), bound)
    decreases t
{
    reveal_with_fuel(forall_tree, 3);
    match t {
        Tree::E => {}
        Tree::T { left, key, value: _, right } => {
            if k < key {
                insert_all_lt(k, v, *left, bound);
            } else if k > key {
                insert_all_lt(k, v, *right, bound);
            }
        }
    }
}

// Helper: insert preserves all_gt
pub proof fn insert_all_gt(k: nat, v: nat, t: Tree, bound: nat)
    requires all_gt(t, bound), k > bound
    ensures all_gt(insert(k, v, t), bound)
    decreases t
{
    reveal_with_fuel(forall_tree, 3);
    match t {
        Tree::E => {}
        Tree::T { left, key, value: _, right } => {
            if k < key {
                insert_all_gt(k, v, *left, bound);
            } else if k > key {
                insert_all_gt(k, v, *right, bound);
            }
        }
    }
}

// Insert preserves BST invariant
pub proof fn insert_bst(k: nat, v: nat, t: Tree)
    requires is_bst(t)
    ensures is_bst(insert(k, v, t))
    decreases t
{
    reveal_with_fuel(is_bst, 2);
    reveal_with_fuel(forall_tree, 2);
    match t {
        Tree::E => {}
        Tree::T { left, key, value: _, right } => {
            if k < key {
                insert_bst(k, v, *left);
                insert_all_lt(k, v, *left, key);
            } else if k > key {
                insert_bst(k, v, *right);
                insert_all_gt(k, v, *right, key);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_empty_bst()
{
    assert(is_bst(Tree::E));
}

pub proof fn example_singleton_bst()
{
    let t = Tree::T {
        left: Box::new(Tree::E),
        key: 5,
        value: 50,
        right: Box::new(Tree::E),
    };
    reveal_with_fuel(is_bst, 2);
    reveal_with_fuel(forall_tree, 2);
    assert(is_bst(t));
}

pub proof fn example_insert_bst()
{
    let t = insert(5, 50, Tree::E);
    insert_bst(5, 50, Tree::E);
    assert(is_bst(t));

    let t2 = insert(3, 30, t);
    insert_bst(3, 30, t);
    assert(is_bst(t2));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn bst_invariant_verify()
{
    example_empty_bst();
    example_singleton_bst();
    example_insert_bst();

    // Build a BST and verify invariant
    let t1 = insert(10, 100, Tree::E);
    insert_bst(10, 100, Tree::E);

    let t2 = insert(5, 50, t1);
    insert_bst(5, 50, t1);

    let t3 = insert(15, 150, t2);
    insert_bst(15, 150, t2);

    assert(is_bst(t3));
}

pub fn main() {
    proof {
        bst_invariant_verify();
    }
}

} // verus!
