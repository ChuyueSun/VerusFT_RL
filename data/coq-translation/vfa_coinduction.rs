use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Coinductive Structures (Supporting VFA)
// ============================================================================

// Coinductive streams (infinite sequences)
pub open spec fn stream_nth(gen: spec_fn(nat) -> nat, n: nat) -> nat { gen(n) }
pub open spec fn stream_take(gen: spec_fn(nat) -> nat, n: nat) -> Seq<nat> { Seq::new(n, |i: int| gen(i as nat)) }

// Constant stream
pub open spec fn const_stream(x: nat) -> spec_fn(nat) -> nat { |_n: nat| x }

// Natural numbers stream
pub open spec fn nats_stream() -> spec_fn(nat) -> nat { |n: nat| n }

// Map over stream
pub open spec fn stream_map(f: spec_fn(nat) -> nat, gen: spec_fn(nat) -> nat) -> spec_fn(nat) -> nat { |n: nat| f(gen(n)) }

// Interleave two streams
pub open spec fn interleave(g1: spec_fn(nat) -> nat, g2: spec_fn(nat) -> nat) -> spec_fn(nat) -> nat {
    |n: nat| if n % 2 == 0 { g1(n / 2) } else { g2(n / 2) }
}

pub proof fn const_stream_constant(x: nat, n: nat) ensures stream_nth(const_stream(x), n) == x {}
pub proof fn nats_stream_nth(n: nat) ensures stream_nth(nats_stream(), n) == n {}
pub proof fn stream_map_nth(f: spec_fn(nat) -> nat, g: spec_fn(nat) -> nat, n: nat)
    ensures stream_nth(stream_map(f, g), n) == f(g(n))
{}

pub proof fn example_stream() { const_stream_constant(42, 10); nats_stream_nth(5); }
pub proof fn coinduction_verify() { example_stream(); }
pub fn main() { proof { coinduction_verify(); } }

} // verus!
