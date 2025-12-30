use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Segment Tree (Supporting VFA)
// ============================================================================

pub enum SegTree { Leaf(nat), Node { sum: nat, left: Box<SegTree>, right: Box<SegTree> } }

pub open spec fn seg_sum(t: SegTree) -> nat {
    match t { SegTree::Leaf(v) => v, SegTree::Node { sum, .. } => sum }
}

pub open spec fn build_seg_tree(s: Seq<nat>, lo: nat, hi: nat) -> SegTree decreases hi - lo {
    if lo + 1 >= hi { SegTree::Leaf(if lo < s.len() { s[lo as int] } else { 0 }) }
    else {
        let mid = (lo + (hi - lo) / 2) as nat;
        let left = build_seg_tree(s, lo, mid);
        let right = build_seg_tree(s, mid, hi);
        SegTree::Node { sum: seg_sum(left) + seg_sum(right), left: Box::new(left), right: Box::new(right) }
    }
}

pub open spec fn query_sum(t: SegTree, lo: nat, hi: nat, qlo: nat, qhi: nat) -> nat decreases t {
    if qlo >= hi || qhi <= lo { 0 }
    else if qlo <= lo && hi <= qhi { seg_sum(t) }
    else { match t {
        SegTree::Leaf(v) => v,
        SegTree::Node { left, right, .. } => {
            let mid = (lo + (hi - lo) / 2) as nat;
            query_sum(*left, lo, mid, qlo, qhi) + query_sum(*right, mid, hi, qlo, qhi)
        }
    }}
}

pub proof fn example_seg_tree() { reveal_with_fuel(build_seg_tree, 3); }
pub proof fn segment_tree_verify() { example_seg_tree(); }
pub fn main() { proof { segment_tree_verify(); } }

} // verus!
