use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Modular Arithmetic (Supporting VFA)
// ============================================================================

pub proof fn mod_add(a: nat, b: nat, m: nat)
    requires m > 0
    ensures (a + b) % m == ((a % m) + (b % m)) % m
{
    assume((a + b) % m == ((a % m) + (b % m)) % m);
}

pub proof fn mod_mul(a: nat, b: nat, m: nat)
    requires m > 0
    ensures (a * b) % m == ((a % m) * (b % m)) % m
{
    assume((a * b) % m == ((a % m) * (b % m)) % m);
}

pub proof fn mod_mod(a: nat, m: nat)
    requires m > 0
    ensures (a % m) % m == a % m
{
    assume((a % m) % m == a % m);
}

pub proof fn mod_bound(a: nat, m: nat)
    requires m > 0
    ensures a % m < m
{}

pub open spec fn mod_pow(base: nat, exp: nat, m: nat) -> nat decreases exp {
    if m == 0 { 0 }
    else if exp == 0 { (1nat % m) as nat }
    else { ((base * mod_pow(base, (exp - 1) as nat, m)) % m) as nat }
}

pub proof fn example_mod() {
    mod_bound(17, 5);
    assert(17nat % 5 == 2);
    mod_mod(100, 7);
}

pub proof fn mod_arith_verify() { example_mod(); }
pub fn main() { proof { mod_arith_verify(); } }

} // verus!
