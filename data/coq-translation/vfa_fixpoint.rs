use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Fixpoint Computations (Supporting VFA)
// ============================================================================

// Iterative fixpoint computation
pub open spec fn iterate_until_fixed(f: spec_fn(nat) -> nat, x: nat, fuel: nat) -> nat decreases fuel {
    if fuel == 0 { x }
    else if f(x) == x { x }
    else { iterate_until_fixed(f, f(x), (fuel - 1) as nat) }
}

// Newton-Raphson style iteration for integer sqrt
pub open spec fn sqrt_step(n: nat, guess: nat) -> nat {
    if guess == 0 { n } else { (guess + n / guess) / 2 }
}

pub open spec fn int_sqrt(n: nat, fuel: nat) -> nat { iterate_until_fixed(|g: nat| sqrt_step(n, g), n, fuel) }

// Least fixpoint exists when function is monotonic on finite lattice
pub open spec fn is_fixpoint(f: spec_fn(nat) -> nat, x: nat) -> bool { f(x) == x }

pub proof fn zero_fixpoint_id() ensures is_fixpoint(|x: nat| x, 0) {}

pub proof fn example_fixpoint() {
    reveal_with_fuel(iterate_until_fixed, 5);
    zero_fixpoint_id();
}

pub proof fn fixpoint_verify() { example_fixpoint(); }
pub fn main() { proof { fixpoint_verify(); } }

} // verus!
