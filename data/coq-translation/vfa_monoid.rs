use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Monoid Structures (Supporting VFA)
// ============================================================================

// Addition monoid on nat
pub proof fn add_assoc(a: nat, b: nat, c: nat) ensures (a + b) + c == a + (b + c) {}
pub proof fn add_zero_left(a: nat) ensures 0 + a == a {}
pub proof fn add_zero_right(a: nat) ensures a + 0 == a {}

// Multiplication monoid on nat
pub proof fn mul_assoc(a: nat, b: nat, c: nat) ensures (a * b) * c == a * (b * c) {
    assert((a * b) * c == a * (b * c)) by (nonlinear_arith);
}
pub proof fn mul_one_left(a: nat) ensures 1 * a == a {}
pub proof fn mul_one_right(a: nat) ensures a * 1 == a {}

// Max monoid on nat
pub open spec fn max(a: nat, b: nat) -> nat { if a > b { a } else { b } }
pub proof fn max_assoc(a: nat, b: nat, c: nat) ensures max(max(a, b), c) == max(a, max(b, c)) {}
pub proof fn max_zero_left(a: nat) ensures max(0, a) == a {}
pub proof fn max_zero_right(a: nat) ensures max(a, 0) == a {}

// Fold with monoid
pub open spec fn fold_add(s: Seq<nat>) -> nat decreases s.len() {
    if s.len() == 0 { 0 } else { s[0] + fold_add(s.skip(1)) }
}

pub open spec fn fold_mul(s: Seq<nat>) -> nat decreases s.len() {
    if s.len() == 0 { 1 } else { s[0] * fold_mul(s.skip(1)) }
}

pub proof fn example_monoid() { add_assoc(1, 2, 3); mul_assoc(2, 3, 4); max_assoc(5, 10, 15); }
pub proof fn monoid_verify() { example_monoid(); }
pub fn main() { proof { monoid_verify(); } }

} // verus!
