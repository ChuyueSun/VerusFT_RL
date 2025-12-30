use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Interval Tree (Supporting VFA)
// ============================================================================

pub struct Interval { pub lo: nat, pub hi: nat }
pub enum IntervalTree {
    E,
    T { interval: Interval, max: nat, left: Box<IntervalTree>, right: Box<IntervalTree> },
}

pub open spec fn interval_valid(i: Interval) -> bool { i.lo <= i.hi }
pub open spec fn overlaps(a: Interval, b: Interval) -> bool { a.lo <= b.hi && b.lo <= a.hi }

pub open spec fn it_max(t: IntervalTree) -> nat {
    match t { IntervalTree::E => 0, IntervalTree::T { max, .. } => max }
}

pub open spec fn search_overlap(t: IntervalTree, q: Interval) -> bool decreases t {
    match t {
        IntervalTree::E => false,
        IntervalTree::T { interval, left, right, .. } =>
            overlaps(interval, q) || search_overlap(*left, q) || search_overlap(*right, q)
    }
}

pub proof fn no_overlap_empty(q: Interval) ensures !search_overlap(IntervalTree::E, q) {}

pub proof fn example_interval() {
    let i1 = Interval { lo: 5, hi: 10 };
    let i2 = Interval { lo: 8, hi: 15 };
    assert(overlaps(i1, i2));
    no_overlap_empty(i1);
}

pub proof fn interval_tree_verify() { example_interval(); }
pub fn main() { proof { interval_tree_verify(); } }

} // verus!
