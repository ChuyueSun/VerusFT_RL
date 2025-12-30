use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Lattice Structures (Supporting VFA)
// ============================================================================

// Lattice operations for nat with min/max
pub open spec fn meet(a: nat, b: nat) -> nat { if a < b { a } else { b } }
pub open spec fn join(a: nat, b: nat) -> nat { if a > b { a } else { b } }

pub proof fn meet_comm(a: nat, b: nat) ensures meet(a, b) == meet(b, a) {}
pub proof fn join_comm(a: nat, b: nat) ensures join(a, b) == join(b, a) {}
pub proof fn meet_assoc(a: nat, b: nat, c: nat) ensures meet(meet(a, b), c) == meet(a, meet(b, c)) {}
pub proof fn join_assoc(a: nat, b: nat, c: nat) ensures join(join(a, b), c) == join(a, join(b, c)) {}

// Absorption laws
pub proof fn absorption_meet(a: nat, b: nat) ensures meet(a, join(a, b)) == a {}
pub proof fn absorption_join(a: nat, b: nat) ensures join(a, meet(a, b)) == a {}

// Idempotence
pub proof fn meet_idemp(a: nat) ensures meet(a, a) == a {}
pub proof fn join_idemp(a: nat) ensures join(a, a) == a {}

// Distributivity
pub proof fn meet_distr_join(a: nat, b: nat, c: nat) ensures meet(a, join(b, c)) == join(meet(a, b), meet(a, c)) {}

pub proof fn example_lattice() { meet_comm(3, 7); join_comm(3, 7); absorption_meet(5, 10); }
pub proof fn lattice_verify() { example_lattice(); }
pub fn main() { proof { lattice_verify(); } }

} // verus!
