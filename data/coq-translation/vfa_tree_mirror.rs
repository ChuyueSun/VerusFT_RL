use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Tree Mirror (Supporting VFA)
// ============================================================================

pub enum Tree { E, T { left: Box<Tree>, value: nat, right: Box<Tree> } }

pub open spec fn mirror(t: Tree) -> Tree decreases t {
    match t {
        Tree::E => Tree::E,
        Tree::T { left, value, right } => Tree::T { left: Box::new(mirror(*right)), value, right: Box::new(mirror(*left)) }
    }
}

pub open spec fn tree_size(t: Tree) -> nat decreases t {
    match t { Tree::E => 0, Tree::T { left, right, .. } => 1 + tree_size(*left) + tree_size(*right) }
}

pub open spec fn tree_height(t: Tree) -> nat decreases t {
    match t { Tree::E => 0, Tree::T { left, right, .. } => 1 + if tree_height(*left) > tree_height(*right) { tree_height(*left) } else { tree_height(*right) } }
}

pub proof fn mirror_mirror(t: Tree) ensures mirror(mirror(t)) == t decreases t {
    reveal_with_fuel(mirror, 2);
    match t { Tree::E => {} Tree::T { left, right, .. } => { mirror_mirror(*left); mirror_mirror(*right); } }
}

pub proof fn mirror_size(t: Tree) ensures tree_size(mirror(t)) == tree_size(t) decreases t {
    reveal_with_fuel(mirror, 2); reveal_with_fuel(tree_size, 2);
    match t { Tree::E => {} Tree::T { left, right, .. } => { mirror_size(*left); mirror_size(*right); } }
}

pub proof fn mirror_height(t: Tree) ensures tree_height(mirror(t)) == tree_height(t) decreases t {
    reveal_with_fuel(mirror, 2); reveal_with_fuel(tree_height, 2);
    match t { Tree::E => {} Tree::T { left, right, .. } => { mirror_height(*left); mirror_height(*right); } }
}

pub proof fn example_mirror() { reveal_with_fuel(mirror, 3); mirror_mirror(Tree::E); }
pub proof fn tree_mirror_verify() { example_mirror(); }
pub fn main() { proof { tree_mirror_verify(); } }

} // verus!
