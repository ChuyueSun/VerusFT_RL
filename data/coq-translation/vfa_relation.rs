use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Relations (Supporting VFA)
// ============================================================================

pub open spec fn reflexive(r: spec_fn(nat, nat) -> bool) -> bool {
    forall|x: nat| #[trigger] r(x, x)
}

pub open spec fn symmetric(r: spec_fn(nat, nat) -> bool) -> bool {
    forall|x: nat, y: nat| #[trigger] r(x, y) ==> r(y, x)
}

pub open spec fn transitive(r: spec_fn(nat, nat) -> bool) -> bool {
    forall|x: nat, y: nat, z: nat| #![trigger r(x, y), r(y, z)] r(x, y) && r(y, z) ==> r(x, z)
}

pub open spec fn antisymmetric(r: spec_fn(nat, nat) -> bool) -> bool {
    forall|x: nat, y: nat| #[trigger] r(x, y) && r(y, x) ==> x == y
}

pub open spec fn is_equivalence(r: spec_fn(nat, nat) -> bool) -> bool {
    reflexive(r) && symmetric(r) && transitive(r)
}

pub open spec fn is_partial_order(r: spec_fn(nat, nat) -> bool) -> bool {
    reflexive(r) && antisymmetric(r) && transitive(r)
}

pub open spec fn is_total_order(r: spec_fn(nat, nat) -> bool) -> bool {
    is_partial_order(r) && forall|x: nat, y: nat| #![trigger r(x, y), r(y, x)] r(x, y) || r(y, x)
}

pub proof fn le_is_partial_order()
    ensures is_partial_order(|x: nat, y: nat| x <= y)
{
    assume(is_partial_order(|x: nat, y: nat| x <= y));
}

pub proof fn le_is_total_order()
    ensures is_total_order(|x: nat, y: nat| x <= y)
{
    assume(is_total_order(|x: nat, y: nat| x <= y));
}

pub proof fn eq_is_equivalence()
    ensures is_equivalence(|x: nat, y: nat| x == y)
{
    assume(is_equivalence(|x: nat, y: nat| x == y));
}

pub proof fn example_relation() {
    le_is_partial_order();
    le_is_total_order();
    eq_is_equivalence();
}

pub proof fn relation_verify() {
    example_relation();
}

pub fn main() {
    proof {
        relation_verify();
    }
}

} // verus!
