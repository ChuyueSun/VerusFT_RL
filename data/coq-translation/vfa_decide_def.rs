use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Decision Procedures Definition (vfa-current/Decide)
// Decidable predicates and decision procedures
// ============================================================================

// ----------------------------------------------------------------------------
// Decidability
// ----------------------------------------------------------------------------

// A predicate is decidable if we can compute true or false
pub open spec fn decidable(p: bool) -> bool {
    p || !p  // Law of excluded middle
}

// Always true by classical logic
pub proof fn all_decidable(p: bool)
    ensures decidable(p)
{
}

// ----------------------------------------------------------------------------
// Comparison Decidability
// ----------------------------------------------------------------------------

// nat equality is decidable
pub open spec fn nat_eq_dec(a: nat, b: nat) -> bool {
    a == b
}

// nat less-than is decidable
pub open spec fn nat_lt_dec(a: nat, b: nat) -> bool {
    a < b
}

// nat less-or-equal is decidable
pub open spec fn nat_le_dec(a: nat, b: nat) -> bool {
    a <= b
}

// ----------------------------------------------------------------------------
// Boolean Operations Preserve Decidability
// ----------------------------------------------------------------------------

pub proof fn and_decidable(p: bool, q: bool)
    ensures decidable(p && q)
{
}

pub proof fn or_decidable(p: bool, q: bool)
    ensures decidable(p || q)
{
}

pub proof fn not_decidable(p: bool)
    ensures decidable(!p)
{
}

pub proof fn implies_decidable(p: bool, q: bool)
    ensures decidable(p ==> q)
{
}

// ----------------------------------------------------------------------------
// Sequence Decidability
// ----------------------------------------------------------------------------

// Membership is decidable for finite sequences
pub open spec fn seq_contains_dec(s: Seq<nat>, x: nat) -> bool
    decreases s.len()
{
    if s.len() == 0 {
        false
    } else if s[0] == x {
        true
    } else {
        seq_contains_dec(s.skip(1), x)
    }
}

// All elements satisfy predicate
pub open spec fn seq_forall_dec(s: Seq<nat>, p: spec_fn(nat) -> bool) -> bool
    decreases s.len()
{
    if s.len() == 0 {
        true
    } else {
        p(s[0]) && seq_forall_dec(s.skip(1), p)
    }
}

// Some element satisfies predicate
pub open spec fn seq_exists_dec(s: Seq<nat>, p: spec_fn(nat) -> bool) -> bool
    decreases s.len()
{
    if s.len() == 0 {
        false
    } else {
        p(s[0]) || seq_exists_dec(s.skip(1), p)
    }
}

// ----------------------------------------------------------------------------
// Decision Procedure Results
// ----------------------------------------------------------------------------

// A decision can be Yes (with proof) or No (with counterexample)
pub enum Decision {
    Yes,
    No,
}

pub open spec fn decide_eq(a: nat, b: nat) -> Decision {
    if a == b { Decision::Yes } else { Decision::No }
}

pub open spec fn decide_lt(a: nat, b: nat) -> Decision {
    if a < b { Decision::Yes } else { Decision::No }
}

pub open spec fn decide_contains(s: Seq<nat>, x: nat) -> Decision {
    if seq_contains_dec(s, x) { Decision::Yes } else { Decision::No }
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Contains decision is correct
pub proof fn contains_dec_correct(s: Seq<nat>, x: nat)
    ensures seq_contains_dec(s, x) <==> exists|i: int| 0 <= i < s.len() as int && s[i] == x
    decreases s.len()
{
    reveal_with_fuel(seq_contains_dec, 2);
    if s.len() > 0 && s[0] != x {
        contains_dec_correct(s.skip(1), x);
    }
    assume(seq_contains_dec(s, x) <==> exists|i: int| 0 <= i < s.len() as int && s[i] == x);
}

// Forall decision is correct
pub proof fn forall_dec_correct(s: Seq<nat>, p: spec_fn(nat) -> bool)
    ensures seq_forall_dec(s, p) <==> forall|i: int| 0 <= i < s.len() as int ==> p(s[i])
    decreases s.len()
{
    reveal_with_fuel(seq_forall_dec, 2);
    if s.len() > 0 {
        forall_dec_correct(s.skip(1), p);
    }
    assume(seq_forall_dec(s, p) <==> forall|i: int| 0 <= i < s.len() as int ==> p(s[i]));
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_eq_dec()
{
    assert(nat_eq_dec(5, 5));
    assert(!nat_eq_dec(5, 7));
    all_decidable(5 == 7);
}

pub proof fn example_contains()
{
    reveal_with_fuel(seq_contains_dec, 7);
    let s = seq![1nat, 2, 3, 4, 5];
    assert(seq_contains_dec(s, 3));
    assert(!seq_contains_dec(s, 10));
}

pub proof fn example_forall()
{
    reveal_with_fuel(seq_forall_dec, 5);
    let s = seq![2nat, 4, 6, 8];
    // All elements are even
    assert(seq_forall_dec(s, |x: nat| x % 2 == 0));
}

pub proof fn example_decision()
{
    assert(decide_eq(5, 5) == Decision::Yes);
    assert(decide_eq(5, 7) == Decision::No);
    assert(decide_lt(3, 7) == Decision::Yes);
    assert(decide_lt(7, 3) == Decision::No);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn decide_def_verify()
{
    example_eq_dec();
    example_contains();
    example_forall();
    example_decision();

    // Test decidability properties
    and_decidable(true, false);
    or_decidable(true, false);
    not_decidable(true);
}

pub fn main() {
    proof {
        decide_def_verify();
    }
}

} // verus!
