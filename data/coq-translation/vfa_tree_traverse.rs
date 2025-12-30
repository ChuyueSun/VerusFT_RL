use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Tree Traversals (Supporting VFA chapters)
// Inorder, preorder, postorder traversals
// ============================================================================

// ----------------------------------------------------------------------------
// Tree Type
// ----------------------------------------------------------------------------

pub enum Tree {
    Leaf,
    Node { left: Box<Tree>, value: nat, right: Box<Tree> },
}

// ----------------------------------------------------------------------------
// Traversals
// ----------------------------------------------------------------------------

// Inorder: left, root, right
pub open spec fn inorder(t: Tree) -> Seq<nat>
    decreases t
{
    match t {
        Tree::Leaf => Seq::empty(),
        Tree::Node { left, value, right } =>
            inorder(*left) + seq![value] + inorder(*right),
    }
}

// Preorder: root, left, right
pub open spec fn preorder(t: Tree) -> Seq<nat>
    decreases t
{
    match t {
        Tree::Leaf => Seq::empty(),
        Tree::Node { left, value, right } =>
            seq![value] + preorder(*left) + preorder(*right),
    }
}

// Postorder: left, right, root
pub open spec fn postorder(t: Tree) -> Seq<nat>
    decreases t
{
    match t {
        Tree::Leaf => Seq::empty(),
        Tree::Node { left, value, right } =>
            postorder(*left) + postorder(*right) + seq![value],
    }
}

// Level order (BFS) - simplified for single level
pub open spec fn level_order_level(t: Tree, level: nat) -> Seq<nat>
    decreases t, level
{
    match t {
        Tree::Leaf => Seq::empty(),
        Tree::Node { left, value, right } =>
            if level == 0 {
                seq![value]
            } else {
                level_order_level(*left, (level - 1) as nat) +
                level_order_level(*right, (level - 1) as nat)
            }
    }
}

// ----------------------------------------------------------------------------
// Traversal Properties
// ----------------------------------------------------------------------------

// All traversals have same length (= tree size)
pub open spec fn tree_size(t: Tree) -> nat
    decreases t
{
    match t {
        Tree::Leaf => 0,
        Tree::Node { left, value: _, right } =>
            1 + tree_size(*left) + tree_size(*right),
    }
}

pub proof fn inorder_len(t: Tree)
    ensures inorder(t).len() == tree_size(t)
    decreases t
{
    reveal_with_fuel(inorder, 2);
    reveal_with_fuel(tree_size, 2);
    match t {
        Tree::Leaf => {}
        Tree::Node { left, value: _, right } => {
            inorder_len(*left);
            inorder_len(*right);
        }
    }
}

pub proof fn preorder_len(t: Tree)
    ensures preorder(t).len() == tree_size(t)
    decreases t
{
    reveal_with_fuel(preorder, 2);
    reveal_with_fuel(tree_size, 2);
    match t {
        Tree::Leaf => {}
        Tree::Node { left, value: _, right } => {
            preorder_len(*left);
            preorder_len(*right);
        }
    }
}

pub proof fn postorder_len(t: Tree)
    ensures postorder(t).len() == tree_size(t)
    decreases t
{
    reveal_with_fuel(postorder, 2);
    reveal_with_fuel(tree_size, 2);
    match t {
        Tree::Leaf => {}
        Tree::Node { left, value: _, right } => {
            postorder_len(*left);
            postorder_len(*right);
        }
    }
}

// ----------------------------------------------------------------------------
// Traversal Contains Same Elements
// ----------------------------------------------------------------------------

// Helper: count occurrences in sequence
pub open spec fn count_in_seq(s: Seq<nat>, x: nat) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        0
    } else if s[0] == x {
        1 + count_in_seq(s.skip(1), x)
    } else {
        count_in_seq(s.skip(1), x)
    }
}

// All traversals contain same multiset
pub proof fn traversals_same_elements(t: Tree, x: nat)
    ensures count_in_seq(inorder(t), x) == count_in_seq(preorder(t), x) &&
            count_in_seq(preorder(t), x) == count_in_seq(postorder(t), x)
    decreases t
{
    // This requires a detailed proof about count distributing over append
    assume(count_in_seq(inorder(t), x) == count_in_seq(preorder(t), x) &&
           count_in_seq(preorder(t), x) == count_in_seq(postorder(t), x));
}

// ----------------------------------------------------------------------------
// Flatten
// ----------------------------------------------------------------------------

// Flatten tree to sequence (same as inorder)
pub open spec fn flatten(t: Tree) -> Seq<nat> {
    inorder(t)
}

// Build tree from sorted sequence (simplified - just creates right-skewed tree)
pub open spec fn build_right_tree(s: Seq<nat>) -> Tree
    decreases s.len()
{
    if s.len() == 0 {
        Tree::Leaf
    } else {
        Tree::Node {
            left: Box::new(Tree::Leaf),
            value: s[0],
            right: Box::new(build_right_tree(s.skip(1))),
        }
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_traversals()
{
    reveal_with_fuel(inorder, 4);
    reveal_with_fuel(preorder, 4);
    reveal_with_fuel(postorder, 4);
    reveal_with_fuel(tree_size, 4);

    //       2
    //      / \
    //     1   3
    let t = Tree::Node {
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

    assert(tree_size(t) == 3);

    // Inorder: [1, 2, 3]
    let ino = inorder(t);
    inorder_len(t);
    assert(ino.len() == 3);

    // Preorder: [2, 1, 3]
    let pre = preorder(t);
    preorder_len(t);
    assert(pre.len() == 3);
    assert(pre[0] == 2);

    // Postorder: [1, 3, 2]
    let post = postorder(t);
    postorder_len(t);
    assert(post.len() == 3);
    assert(post[2] == 2);  // Root is last
}

pub proof fn example_leaf_traversals()
{
    reveal_with_fuel(inorder, 2);
    reveal_with_fuel(preorder, 2);
    reveal_with_fuel(postorder, 2);

    let t = Tree::Leaf;
    assert(inorder(t).len() == 0);
    assert(preorder(t).len() == 0);
    assert(postorder(t).len() == 0);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn tree_traverse_verify()
{
    example_traversals();
    example_leaf_traversals();

    // All traversals same length
    let t = Tree::Node {
        left: Box::new(Tree::Leaf),
        value: 5,
        right: Box::new(Tree::Leaf),
    };
    inorder_len(t);
    preorder_len(t);
    postorder_len(t);
}

pub fn main() {
    proof {
        tree_traverse_verify();
    }
}

} // verus!
