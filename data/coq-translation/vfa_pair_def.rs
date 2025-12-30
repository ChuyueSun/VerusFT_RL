use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Pair/Tuple Types (Supporting VFA)
// ============================================================================

pub open spec fn fst<A, B>(p: (A, B)) -> A { p.0 }
pub open spec fn snd<A, B>(p: (A, B)) -> B { p.1 }

pub open spec fn swap<A, B>(p: (A, B)) -> (B, A) { (p.1, p.0) }

pub open spec fn map_fst<A, B, C>(p: (A, B), f: spec_fn(A) -> C) -> (C, B) { (f(p.0), p.1) }
pub open spec fn map_snd<A, B, C>(p: (A, B), f: spec_fn(B) -> C) -> (A, C) { (p.0, f(p.1)) }

pub proof fn fst_pair<A, B>(a: A, b: B) ensures fst((a, b)) == a {}
pub proof fn snd_pair<A, B>(a: A, b: B) ensures snd((a, b)) == b {}
pub proof fn swap_swap<A, B>(p: (A, B)) ensures swap(swap(p)) == p {}

pub proof fn example_pair()
{
    let p = (3nat, 5nat);
    fst_pair(3nat, 5nat);
    snd_pair(3nat, 5nat);
    assert(fst(p) == 3);
    assert(snd(p) == 5);
    swap_swap(p);
}

pub proof fn pair_def_verify() { example_pair(); }
pub fn main() { proof { pair_def_verify(); } }

} // verus!
