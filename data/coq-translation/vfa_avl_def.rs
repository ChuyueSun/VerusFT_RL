use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: AVL Tree Definition (Supporting VFA)
// Self-balancing binary search tree
// ============================================================================

pub enum AVL {
    E,
    T { height: nat, left: Box<AVL>, key: int, value: nat, right: Box<AVL> },
}

pub open spec fn avl_height(t: AVL) -> nat {
    match t { AVL::E => 0, AVL::T { height, .. } => height }
}

pub open spec fn balance_factor(t: AVL) -> int {
    match t {
        AVL::E => 0,
        AVL::T { left, right, .. } => avl_height(*right) as int - avl_height(*left) as int
    }
}

pub open spec fn is_balanced(t: AVL) -> bool decreases t {
    match t {
        AVL::E => true,
        AVL::T { left, right, .. } =>
            -1 <= balance_factor(t) <= 1 && is_balanced(*left) && is_balanced(*right)
    }
}

pub open spec fn make_node(l: AVL, k: int, v: nat, r: AVL) -> AVL {
    let h = if avl_height(l) > avl_height(r) { avl_height(l) } else { avl_height(r) };
    AVL::T { height: h + 1, left: Box::new(l), key: k, value: v, right: Box::new(r) }
}

pub proof fn example_avl() {
    reveal_with_fuel(is_balanced, 3);
    assert(is_balanced(AVL::E));
    let t = make_node(AVL::E, 5, 50, AVL::E);
    assert(balance_factor(t) == 0);
}

pub proof fn avl_def_verify() { example_avl(); }
pub fn main() { proof { avl_def_verify(); } }

} // verus!
