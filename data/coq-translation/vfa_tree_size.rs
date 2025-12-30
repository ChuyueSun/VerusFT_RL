use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Tree Size Properties (Supporting VFA chapters)
// Size-related properties of binary trees
// ============================================================================

// ----------------------------------------------------------------------------
// Tree Type
// ----------------------------------------------------------------------------

pub enum Tree<T> {
    Leaf,
    Node { left: Box<Tree<T>>, value: T, right: Box<Tree<T>> },
}

// ----------------------------------------------------------------------------
// Size Functions
// ----------------------------------------------------------------------------

// Number of nodes
pub open spec fn tree_size<T>(t: Tree<T>) -> nat
    decreases t
{
    match t {
        Tree::Leaf => 0,
        Tree::Node { left, value: _, right } =>
            1 + tree_size(*left) + tree_size(*right),
    }
}

// Number of leaves
pub open spec fn tree_leaves<T>(t: Tree<T>) -> nat
    decreases t
{
    match t {
        Tree::Leaf => 1,
        Tree::Node { left, value: _, right } =>
            tree_leaves(*left) + tree_leaves(*right),
    }
}

// Tree height
pub open spec fn tree_height<T>(t: Tree<T>) -> nat
    decreases t
{
    match t {
        Tree::Leaf => 0,
        Tree::Node { left, value: _, right } => {
            let lh = tree_height(*left);
            let rh = tree_height(*right);
            1 + if lh > rh { lh } else { rh }
        }
    }
}

// ----------------------------------------------------------------------------
// Size Properties
// ----------------------------------------------------------------------------

// Size is non-negative (trivially true for nat)
pub proof fn size_nonneg<T>(t: Tree<T>)
    ensures tree_size(t) >= 0
{
}

// Leaves = nodes + 1
pub proof fn leaves_nodes_relation<T>(t: Tree<T>)
    ensures tree_leaves(t) == tree_size(t) + 1
    decreases t
{
    reveal_with_fuel(tree_size, 2);
    reveal_with_fuel(tree_leaves, 2);
    match t {
        Tree::Leaf => {}
        Tree::Node { left, value: _, right } => {
            leaves_nodes_relation(*left);
            leaves_nodes_relation(*right);
        }
    }
}

// Height bounds size: size < 2^height
pub proof fn height_bounds_size<T>(t: Tree<T>)
    ensures tree_size(t) < pow2(tree_height(t))
    decreases t
{
    reveal_with_fuel(tree_size, 2);
    reveal_with_fuel(tree_height, 2);
    match t {
        Tree::Leaf => {
            assert(tree_size(t) == 0);
            assert(tree_height(t) == 0);
            pow2_pos(0);
        }
        Tree::Node { left, value: _, right } => {
            height_bounds_size(*left);
            height_bounds_size(*right);
            let lh = tree_height(*left);
            let rh = tree_height(*right);
            let h = if lh > rh { lh } else { rh };
            pow2_monotonic(lh, h);
            pow2_monotonic(rh, h);
            pow2_double(h);
        }
    }
}

// Helper: 2^n
pub open spec fn pow2(n: nat) -> nat
    decreases n
{
    if n == 0 { 1 } else { 2 * pow2((n - 1) as nat) }
}

pub proof fn pow2_pos(n: nat)
    ensures pow2(n) > 0
    decreases n
{
    reveal_with_fuel(pow2, 2);
    if n > 0 {
        pow2_pos((n - 1) as nat);
    }
}

pub proof fn pow2_monotonic(a: nat, b: nat)
    requires a <= b
    ensures pow2(a) <= pow2(b)
    decreases b
{
    reveal_with_fuel(pow2, 2);
    if a < b {
        pow2_monotonic(a, (b - 1) as nat);
    }
}

pub proof fn pow2_double(n: nat)
    ensures pow2(n) + pow2(n) == pow2(n + 1)
{
    reveal_with_fuel(pow2, 2);
}

// ----------------------------------------------------------------------------
// Perfect Trees
// ----------------------------------------------------------------------------

// A tree is perfect if all leaves are at the same depth
pub open spec fn is_perfect<T>(t: Tree<T>) -> bool
    decreases t
{
    match t {
        Tree::Leaf => true,
        Tree::Node { left, value: _, right } =>
            tree_height(*left) == tree_height(*right) &&
            is_perfect(*left) &&
            is_perfect(*right),
    }
}

// Perfect tree has exactly 2^h - 1 nodes
pub proof fn perfect_size<T>(t: Tree<T>)
    requires is_perfect(t)
    ensures tree_size(t) == pow2(tree_height(t)) - 1
    decreases t
{
    reveal_with_fuel(tree_size, 2);
    reveal_with_fuel(tree_height, 2);
    reveal_with_fuel(is_perfect, 2);
    reveal_with_fuel(pow2, 2);
    match t {
        Tree::Leaf => {}
        Tree::Node { left, value: _, right } => {
            perfect_size(*left);
            perfect_size(*right);
            let h = tree_height(*left);
            // size = 1 + (2^h - 1) + (2^h - 1) = 2^(h+1) - 1
            pow2_double(h);
        }
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_leaf()
{
    let t: Tree<nat> = Tree::Leaf;
    assert(tree_size(t) == 0);
    assert(tree_leaves(t) == 1);
    assert(tree_height(t) == 0);
    leaves_nodes_relation(t);
}

pub proof fn example_single_node()
{
    reveal_with_fuel(tree_size, 3);
    reveal_with_fuel(tree_leaves, 3);
    reveal_with_fuel(tree_height, 3);

    let t: Tree<nat> = Tree::Node {
        left: Box::new(Tree::Leaf),
        value: 5,
        right: Box::new(Tree::Leaf),
    };
    assert(tree_size(t) == 1);
    assert(tree_leaves(t) == 2);
    assert(tree_height(t) == 1);
    leaves_nodes_relation(t);
}

pub proof fn example_perfect()
{
    reveal_with_fuel(tree_size, 4);
    reveal_with_fuel(tree_height, 4);
    reveal_with_fuel(is_perfect, 4);
    reveal_with_fuel(pow2, 3);

    // Perfect tree of height 2
    let t: Tree<nat> = Tree::Node {
        left: Box::new(Tree::Node {
            left: Box::new(Tree::Leaf),
            value: 1,
            right: Box::new(Tree::Leaf),
        }),
        value: 2,
        right: Box::new(Tree::Node {
            left: Box::new(Tree::Leaf),
            value: 3,
            right: Box::new(Tree::Leaf),
        }),
    };

    assert(tree_height(t) == 2);
    assert(tree_size(t) == 3);
    assert(is_perfect(t));
    perfect_size(t);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn tree_size_verify()
{
    example_leaf();
    example_single_node();
    example_perfect();

    // Test pow2
    pow2_pos(5);
    pow2_monotonic(3, 5);
}

pub fn main() {
    proof {
        tree_size_verify();
    }
}

} // verus!
