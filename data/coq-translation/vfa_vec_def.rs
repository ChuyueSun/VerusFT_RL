use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Vector Operations (Supporting VFA)
// ============================================================================

pub open spec fn vec_add(v1: Seq<nat>, v2: Seq<nat>) -> Seq<nat>
    recommends v1.len() == v2.len()
{ Seq::new(v1.len(), |i: int| v1[i] + v2[i]) }

pub open spec fn scalar_mul(k: nat, v: Seq<nat>) -> Seq<nat>
{ Seq::new(v.len(), |i: int| k * v[i]) }

pub open spec fn dot_product(v1: Seq<nat>, v2: Seq<nat>) -> nat
    recommends v1.len() == v2.len()
    decreases v1.len()
{
    if v1.len() == 0 { 0 }
    else { v1[0] * v2[0] + dot_product(v1.skip(1), v2.skip(1)) }
}

pub proof fn vec_add_comm(v1: Seq<nat>, v2: Seq<nat>)
    requires v1.len() == v2.len()
    ensures vec_add(v1, v2) =~= vec_add(v2, v1)
{}

pub proof fn scalar_mul_zero(v: Seq<nat>) ensures scalar_mul(0, v) =~= Seq::new(v.len(), |_i: int| 0nat) {}
pub proof fn scalar_mul_one(v: Seq<nat>) ensures scalar_mul(1, v) =~= v {}

pub proof fn example_vec() {
    let v1 = seq![1nat, 2, 3];
    let v2 = seq![4nat, 5, 6];
    vec_add_comm(v1, v2);
    reveal_with_fuel(dot_product, 4);
    assert(dot_product(v1, v2) == 1*4 + 2*5 + 3*6);
}

pub proof fn vec_def_verify() { example_vec(); }
pub fn main() { proof { vec_def_verify(); } }

} // verus!
