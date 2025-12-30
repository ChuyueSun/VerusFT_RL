use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Tree Balance Properties (Supporting VFA)
// ============================================================================

pub enum Tree { E, T { left: Box<Tree>, value: nat, right: Box<Tree> } }

pub open spec fn height(t: Tree) -> nat decreases t {
    match t { Tree::E => 0, Tree::T { left, right, .. } => 1 + max(height(*left), height(*right)) }
}

pub open spec fn max(a: nat, b: nat) -> nat { if a > b { a } else { b } }
pub open spec fn abs_diff(a: nat, b: nat) -> nat { if a > b { (a - b) as nat } else { (b - a) as nat } }

pub open spec fn is_balanced(t: Tree) -> bool decreases t {
    match t {
        Tree::E => true,
        Tree::T { left, right, .. } =>
            abs_diff(height(*left), height(*right)) <= 1 && is_balanced(*left) && is_balanced(*right)
    }
}

pub open spec fn is_complete(t: Tree) -> bool decreases t {
    match t {
        Tree::E => true,
        Tree::T { left, right, .. } =>
            abs_diff(height(*left), height(*right)) == 0 && is_complete(*left) && is_complete(*right)
    }
}

pub proof fn complete_implies_balanced(t: Tree) requires is_complete(t) ensures is_balanced(t) decreases t {
    reveal_with_fuel(is_complete, 2); reveal_with_fuel(is_balanced, 2);
    match t { Tree::E => {} Tree::T { left, right, .. } => { complete_implies_balanced(*left); complete_implies_balanced(*right); } }
}

pub proof fn example_balance() { reveal_with_fuel(is_balanced, 2); assert(is_balanced(Tree::E)); }
pub proof fn tree_balance_verify() { example_balance(); }
pub fn main() { proof { tree_balance_verify(); } }

} // verus!
