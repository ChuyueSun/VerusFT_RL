use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Treap (Tree + Heap) (Supporting VFA)
// ============================================================================

pub enum Treap {
    E,
    T { key: int, priority: nat, left: Box<Treap>, right: Box<Treap> },
}

pub open spec fn treap_key(t: Treap) -> Option<int> {
    match t { Treap::E => None, Treap::T { key, .. } => Some(key) }
}

pub open spec fn treap_priority(t: Treap) -> Option<nat> {
    match t { Treap::E => None, Treap::T { priority, .. } => Some(priority) }
}

// BST property on keys
pub open spec fn is_bst_treap(t: Treap) -> bool decreases t {
    match t {
        Treap::E => true,
        Treap::T { key, left, right, .. } =>
            forall_lt(*left, key) && forall_gt(*right, key) && is_bst_treap(*left) && is_bst_treap(*right)
    }
}

pub open spec fn forall_lt(t: Treap, bound: int) -> bool decreases t {
    match t { Treap::E => true, Treap::T { key, left, right, .. } => key < bound && forall_lt(*left, bound) && forall_lt(*right, bound) }
}

pub open spec fn forall_gt(t: Treap, bound: int) -> bool decreases t {
    match t { Treap::E => true, Treap::T { key, left, right, .. } => key > bound && forall_gt(*left, bound) && forall_gt(*right, bound) }
}

// Heap property on priorities (max-heap)
pub open spec fn is_heap_treap(t: Treap) -> bool decreases t {
    match t {
        Treap::E => true,
        Treap::T { priority, left, right, .. } =>
            priority_ge(*left, priority) && priority_ge(*right, priority) && is_heap_treap(*left) && is_heap_treap(*right)
    }
}

pub open spec fn priority_ge(t: Treap, bound: nat) -> bool {
    match t { Treap::E => true, Treap::T { priority, .. } => priority <= bound }
}

pub proof fn example_treap() { reveal_with_fuel(is_bst_treap, 2); assert(is_bst_treap(Treap::E)); }
pub proof fn treap_verify() { example_treap(); }
pub fn main() { proof { treap_verify(); } }

} // verus!
