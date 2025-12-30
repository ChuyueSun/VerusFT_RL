use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Bit Operations (Supporting VFA)
// Using u64 for bitwise operations
// ============================================================================

pub open spec fn bit_and(a: u64, b: u64) -> u64 { a & b }
pub open spec fn bit_or(a: u64, b: u64) -> u64 { a | b }
pub open spec fn bit_xor(a: u64, b: u64) -> u64 { a ^ b }

pub proof fn and_comm(a: u64, b: u64) ensures bit_and(a, b) == bit_and(b, a) {
    assert(bit_and(a, b) == bit_and(b, a)) by (bit_vector);
}
pub proof fn or_comm(a: u64, b: u64) ensures bit_or(a, b) == bit_or(b, a) {
    assert(bit_or(a, b) == bit_or(b, a)) by (bit_vector);
}
pub proof fn xor_comm(a: u64, b: u64) ensures bit_xor(a, b) == bit_xor(b, a) {
    assert(bit_xor(a, b) == bit_xor(b, a)) by (bit_vector);
}

pub proof fn xor_self(a: u64) ensures bit_xor(a, a) == 0 {
    assert(bit_xor(a, a) == 0) by (bit_vector);
}
pub proof fn xor_zero(a: u64) ensures bit_xor(a, 0) == a {
    assert(bit_xor(a, 0) == a) by (bit_vector);
}
pub proof fn and_zero(a: u64) ensures bit_and(a, 0) == 0 {
    assert(bit_and(a, 0) == 0) by (bit_vector);
}
pub proof fn or_zero(a: u64) ensures bit_or(a, 0) == a {
    assert(bit_or(a, 0) == a) by (bit_vector);
}

pub open spec fn is_power_of_two(n: u64) -> bool { n > 0 && bit_and(n, (n - 1) as u64) == 0 }

pub proof fn example_bits() {
    xor_self(42);
    assert(bit_xor(42, 42) == 0);
    and_zero(100);
    assert(bit_and(100, 0) == 0);
}

pub proof fn bit_ops_verify() { example_bits(); }
pub fn main() { proof { bit_ops_verify(); } }

} // verus!
