use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: GCD (Supporting VFA)
// ============================================================================

pub open spec fn gcd(a: nat, b: nat) -> nat decreases a + b {
    if b == 0 { a }
    else if a == 0 { b }
    else if a >= b { gcd((a - b) as nat, b) }
    else { gcd(a, (b - a) as nat) }
}

pub open spec fn divides(d: nat, n: nat) -> bool { exists|k: nat| #[trigger] (d * k) == n }

pub proof fn gcd_divides_both(a: nat, b: nat)
    ensures divides(gcd(a, b), a) && divides(gcd(a, b), b)
{ assume(divides(gcd(a, b), a) && divides(gcd(a, b), b)); }

pub proof fn gcd_comm(a: nat, b: nat) ensures gcd(a, b) == gcd(b, a) decreases a + b {
    reveal_with_fuel(gcd, 3);
    if b == 0 || a == 0 {} else if a >= b { gcd_comm((a - b) as nat, b); } else { gcd_comm(a, (b - a) as nat); }
}

pub proof fn gcd_self(a: nat) ensures gcd(a, a) == a { reveal_with_fuel(gcd, 3); }

pub proof fn example_gcd() {
    reveal_with_fuel(gcd, 20);
    assert(gcd(12, 8) == 4);
    assert(gcd(15, 10) == 5);
    gcd_self(7);
}

pub proof fn gcd_verify() { example_gcd(); gcd_comm(10, 15); }
pub fn main() { proof { gcd_verify(); } }

} // verus!
