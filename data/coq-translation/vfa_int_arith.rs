use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Integer Arithmetic (Supporting VFA)
// ============================================================================

pub proof fn add_comm_int(a: int, b: int) ensures a + b == b + a {}
pub proof fn add_assoc_int(a: int, b: int, c: int) ensures (a + b) + c == a + (b + c) {}
pub proof fn mul_comm_int(a: int, b: int) ensures a * b == b * a {}
pub proof fn mul_assoc_int(a: int, b: int, c: int) ensures (a * b) * c == a * (b * c) {
    assert((a * b) * c == a * (b * c)) by (nonlinear_arith);
}
pub proof fn distr_int(a: int, b: int, c: int) ensures a * (b + c) == a * b + a * c {
    assert(a * (b + c) == a * b + a * c) by (nonlinear_arith);
}

pub proof fn neg_neg(a: int) ensures -(-a) == a {}
pub proof fn add_neg(a: int) ensures a + (-a) == 0 {}
pub proof fn sub_def(a: int, b: int) ensures a - b == a + (-b) {}

pub open spec fn abs(x: int) -> nat { if x >= 0 { x as nat } else { (-x) as nat } }
pub proof fn abs_nonneg(x: int) ensures abs(x) >= 0 {}
pub proof fn abs_neg(x: int) ensures abs(-x) == abs(x) {}

pub proof fn example_int_arith() {
    add_comm_int(3, -5);
    neg_neg(-7);
    abs_neg(10);
}

pub proof fn int_arith_verify() { example_int_arith(); }
pub fn main() { proof { int_arith_verify(); } }

} // verus!
