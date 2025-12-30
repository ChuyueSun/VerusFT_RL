use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Tree Flatten (Supporting VFA)
// ============================================================================

pub enum Tree { E, T { left: Box<Tree>, value: nat, right: Box<Tree> } }

pub open spec fn flatten(t: Tree) -> Seq<nat> decreases t {
    match t {
        Tree::E => Seq::empty(),
        Tree::T { left, value, right } => flatten(*left) + seq![value] + flatten(*right)
    }
}

pub open spec fn tree_size(t: Tree) -> nat decreases t {
    match t { Tree::E => 0, Tree::T { left, right, .. } => 1 + tree_size(*left) + tree_size(*right) }
}

pub proof fn flatten_len(t: Tree) ensures flatten(t).len() == tree_size(t) decreases t {
    reveal_with_fuel(flatten, 2); reveal_with_fuel(tree_size, 2);
    match t { Tree::E => {} Tree::T { left, right, .. } => { flatten_len(*left); flatten_len(*right); } }
}

pub open spec fn tree_from_seq(s: Seq<nat>) -> Tree decreases s.len() {
    if s.len() == 0 { Tree::E }
    else { Tree::T { left: Box::new(Tree::E), value: s[0], right: Box::new(tree_from_seq(s.skip(1))) } }
}

pub proof fn example_flatten() {
    reveal_with_fuel(flatten, 4);
    reveal_with_fuel(tree_size, 4);
    let t = Tree::T { left: Box::new(Tree::E), value: 5, right: Box::new(Tree::E) };
    flatten_len(t);
    assert(flatten(t).len() == 1);
}

pub proof fn tree_flatten_verify() { example_flatten(); }
pub fn main() { proof { tree_flatten_verify(); } }

} // verus!
