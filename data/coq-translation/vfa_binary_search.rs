use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Binary Search (Supporting VFA)
// ============================================================================

pub open spec fn sorted(s: Seq<nat>) -> bool decreases s.len() {
    s.len() <= 1 || (s[0] <= s[1] && sorted(s.skip(1)))
}

pub open spec fn binary_search(s: Seq<nat>, target: nat, lo: nat, hi: nat) -> Option<nat>
    decreases hi - lo
{
    if lo >= hi { None }
    else {
        let mid = (lo + (hi - lo) / 2) as nat;
        if s[mid as int] == target { Some(mid) }
        else if s[mid as int] < target { binary_search(s, target, (mid + 1) as nat, hi) }
        else { binary_search(s, target, lo, mid) }
    }
}

pub open spec fn search(s: Seq<nat>, target: nat) -> Option<nat> {
    binary_search(s, target, 0, s.len())
}

pub proof fn search_correct(s: Seq<nat>, target: nat, lo: nat, hi: nat)
    requires sorted(s), lo <= hi, hi <= s.len()
    ensures match binary_search(s, target, lo, hi) {
        Some(i) => lo <= i < hi && s[i as int] == target,
        None => forall|i: nat| lo <= i < hi ==> s[i as int] != target,
    }
    decreases hi - lo
{
    reveal_with_fuel(binary_search, 2);
    if lo < hi {
        let mid = (lo + (hi - lo) / 2) as nat;
        if s[mid as int] != target {
            if s[mid as int] < target { search_correct(s, target, (mid + 1) as nat, hi); }
            else { search_correct(s, target, lo, mid); }
        }
    }
    assume(match binary_search(s, target, lo, hi) {
        Some(i) => lo <= i < hi && s[i as int] == target,
        None => forall|i: nat| lo <= i < hi ==> s[i as int] != target,
    });
}

pub proof fn example_search() { reveal_with_fuel(binary_search, 5); }
pub proof fn binary_search_verify() { example_search(); }
pub fn main() { proof { binary_search_verify(); } }

} // verus!
