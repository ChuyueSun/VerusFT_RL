use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Boolean Logic (Supporting VFA)
// Boolean operations and De Morgan's laws
// ============================================================================

// ----------------------------------------------------------------------------
// Basic Boolean Operations
// ----------------------------------------------------------------------------

pub open spec fn bool_and(a: bool, b: bool) -> bool { a && b }
pub open spec fn bool_or(a: bool, b: bool) -> bool { a || b }
pub open spec fn bool_not(a: bool) -> bool { !a }
pub open spec fn bool_xor(a: bool, b: bool) -> bool { a != b }
pub open spec fn bool_implies(a: bool, b: bool) -> bool { !a || b }
pub open spec fn bool_iff(a: bool, b: bool) -> bool { a == b }

// ----------------------------------------------------------------------------
// Identity Laws
// ----------------------------------------------------------------------------

pub proof fn and_true(a: bool) ensures bool_and(a, true) == a {}
pub proof fn and_false(a: bool) ensures bool_and(a, false) == false {}
pub proof fn or_true(a: bool) ensures bool_or(a, true) == true {}
pub proof fn or_false(a: bool) ensures bool_or(a, false) == a {}
pub proof fn not_not(a: bool) ensures bool_not(bool_not(a)) == a {}

// ----------------------------------------------------------------------------
// Commutativity
// ----------------------------------------------------------------------------

pub proof fn and_comm(a: bool, b: bool) ensures bool_and(a, b) == bool_and(b, a) {}
pub proof fn or_comm(a: bool, b: bool) ensures bool_or(a, b) == bool_or(b, a) {}
pub proof fn xor_comm(a: bool, b: bool) ensures bool_xor(a, b) == bool_xor(b, a) {}
pub proof fn iff_comm(a: bool, b: bool) ensures bool_iff(a, b) == bool_iff(b, a) {}

// ----------------------------------------------------------------------------
// Associativity
// ----------------------------------------------------------------------------

pub proof fn and_assoc(a: bool, b: bool, c: bool)
    ensures bool_and(bool_and(a, b), c) == bool_and(a, bool_and(b, c))
{}

pub proof fn or_assoc(a: bool, b: bool, c: bool)
    ensures bool_or(bool_or(a, b), c) == bool_or(a, bool_or(b, c))
{}

// ----------------------------------------------------------------------------
// Distributivity
// ----------------------------------------------------------------------------

pub proof fn and_or_distr(a: bool, b: bool, c: bool)
    ensures bool_and(a, bool_or(b, c)) == bool_or(bool_and(a, b), bool_and(a, c))
{}

pub proof fn or_and_distr(a: bool, b: bool, c: bool)
    ensures bool_or(a, bool_and(b, c)) == bool_and(bool_or(a, b), bool_or(a, c))
{}

// ----------------------------------------------------------------------------
// De Morgan's Laws
// ----------------------------------------------------------------------------

pub proof fn de_morgan_and(a: bool, b: bool)
    ensures bool_not(bool_and(a, b)) == bool_or(bool_not(a), bool_not(b))
{}

pub proof fn de_morgan_or(a: bool, b: bool)
    ensures bool_not(bool_or(a, b)) == bool_and(bool_not(a), bool_not(b))
{}

// ----------------------------------------------------------------------------
// Implication Properties
// ----------------------------------------------------------------------------

pub proof fn implies_def(a: bool, b: bool)
    ensures bool_implies(a, b) == bool_or(bool_not(a), b)
{}

pub proof fn implies_true(a: bool)
    ensures bool_implies(a, true) == true
{}

pub proof fn false_implies(b: bool)
    ensures bool_implies(false, b) == true
{}

pub proof fn implies_self(a: bool)
    ensures bool_implies(a, a) == true
{}

pub proof fn contrapositive(a: bool, b: bool)
    ensures bool_implies(a, b) == bool_implies(bool_not(b), bool_not(a))
{}

// ----------------------------------------------------------------------------
// Idempotence
// ----------------------------------------------------------------------------

pub proof fn and_idemp(a: bool) ensures bool_and(a, a) == a {}
pub proof fn or_idemp(a: bool) ensures bool_or(a, a) == a {}

// ----------------------------------------------------------------------------
// Absorption
// ----------------------------------------------------------------------------

pub proof fn and_absorb(a: bool, b: bool)
    ensures bool_and(a, bool_or(a, b)) == a
{}

pub proof fn or_absorb(a: bool, b: bool)
    ensures bool_or(a, bool_and(a, b)) == a
{}

// ----------------------------------------------------------------------------
// Complement
// ----------------------------------------------------------------------------

pub proof fn and_complement(a: bool)
    ensures bool_and(a, bool_not(a)) == false
{}

pub proof fn or_complement(a: bool)
    ensures bool_or(a, bool_not(a)) == true
{}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_de_morgan()
{
    de_morgan_and(true, false);
    assert(bool_not(bool_and(true, false)) == bool_or(bool_not(true), bool_not(false)));

    de_morgan_or(true, false);
    assert(bool_not(bool_or(true, false)) == bool_and(bool_not(true), bool_not(false)));
}

pub proof fn example_implies()
{
    implies_true(true);
    implies_true(false);
    false_implies(true);
    false_implies(false);

    contrapositive(true, false);
}

pub proof fn example_xor()
{
    assert(bool_xor(true, true) == false);
    assert(bool_xor(true, false) == true);
    assert(bool_xor(false, true) == true);
    assert(bool_xor(false, false) == false);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn logic_bool_verify()
{
    example_de_morgan();
    example_implies();
    example_xor();

    // Test all laws
    and_comm(true, false);
    or_comm(true, false);
    and_assoc(true, false, true);
    and_or_distr(true, false, true);
}

pub fn main() {
    proof {
        logic_bool_verify();
    }
}

} // verus!
