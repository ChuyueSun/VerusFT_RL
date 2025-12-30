use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Range/Interval (Supporting VFA)
// ============================================================================

pub struct Range { pub lo: nat, pub hi: nat }

pub open spec fn range_valid(r: Range) -> bool { r.lo <= r.hi }
pub open spec fn range_len(r: Range) -> nat { if r.lo <= r.hi { (r.hi - r.lo) as nat } else { 0 } }
pub open spec fn in_range(x: nat, r: Range) -> bool { r.lo <= x && x < r.hi }
pub open spec fn range_empty(r: Range) -> bool { r.lo >= r.hi }

pub open spec fn range_intersect(r1: Range, r2: Range) -> Range {
    let lo = if r1.lo > r2.lo { r1.lo } else { r2.lo };
    let hi = if r1.hi < r2.hi { r1.hi } else { r2.hi };
    Range { lo, hi }
}

pub open spec fn range_union_len(r1: Range, r2: Range) -> nat {
    (range_len(r1) + range_len(r2) - range_len(range_intersect(r1, r2))) as nat
}

pub proof fn empty_range_len(r: Range) requires range_empty(r) ensures range_len(r) == 0 {}

pub proof fn example_range() {
    let r = Range { lo: 5, hi: 10 };
    assert(range_len(r) == 5);
    assert(in_range(7, r));
    assert(!in_range(10, r));
}

pub proof fn range_def_verify() { example_range(); }
pub fn main() { proof { range_def_verify(); } }

} // verus!
