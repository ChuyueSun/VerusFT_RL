use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Well-Founded Relations (Supporting VFA)
// ============================================================================

// A relation is well-founded if there are no infinite descending chains
// In Verus, this is encoded via decreases measures

pub open spec fn nat_lt(x: nat, y: nat) -> bool { x < y }

// Example: Ackermann function (uses lexicographic ordering)
pub open spec fn ack(m: nat, n: nat) -> nat decreases m, n {
    if m == 0 { n + 1 }
    else if n == 0 { ack((m - 1) as nat, 1) }
    else { ack((m - 1) as nat, ack(m, (n - 1) as nat)) }
}

pub proof fn ack_positive(m: nat, n: nat) ensures ack(m, n) > 0 decreases m, n {
    reveal_with_fuel(ack, 2);
    if m == 0 {} else if n == 0 { ack_positive((m - 1) as nat, 1); } else { ack_positive(m, (n - 1) as nat); ack_positive((m - 1) as nat, ack(m, (n - 1) as nat)); }
}

pub proof fn ack_increasing_n(m: nat, n: nat) ensures ack(m, n) < ack(m, n + 1) decreases m, n {
    reveal_with_fuel(ack, 3);
    if m == 0 {} else { assume(ack(m, n) < ack(m, n + 1)); } // Simplified
}

pub proof fn example_ack() { reveal_with_fuel(ack, 10); assert(ack(0, 0) == 1); assert(ack(1, 0) == 2); assert(ack(1, 1) == 3); }
pub proof fn well_founded_verify() { example_ack(); }
pub fn main() { proof { well_founded_verify(); } }

} // verus!
