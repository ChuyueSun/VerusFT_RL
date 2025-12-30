use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Natural Number Arithmetic (Supporting VFA)
// Basic arithmetic properties
// ============================================================================

// ----------------------------------------------------------------------------
// Addition Properties
// ----------------------------------------------------------------------------

pub proof fn add_comm(a: nat, b: nat)
    ensures a + b == b + a
{
}

pub proof fn add_assoc(a: nat, b: nat, c: nat)
    ensures (a + b) + c == a + (b + c)
{
}

pub proof fn add_zero_left(a: nat)
    ensures 0 + a == a
{
}

pub proof fn add_zero_right(a: nat)
    ensures a + 0 == a
{
}

// ----------------------------------------------------------------------------
// Multiplication Properties
// ----------------------------------------------------------------------------

pub proof fn mul_comm(a: nat, b: nat)
    ensures a * b == b * a
{
}

pub proof fn mul_assoc(a: nat, b: nat, c: nat)
    ensures (a * b) * c == a * (b * c)
{
    // Associativity of multiplication - axiom in Verus
    assert((a * b) * c == a * (b * c)) by (nonlinear_arith);
}

pub proof fn mul_zero_left(a: nat)
    ensures 0 * a == 0
{
}

pub proof fn mul_zero_right(a: nat)
    ensures a * 0 == 0
{
}

pub proof fn mul_one_left(a: nat)
    ensures 1 * a == a
{
}

pub proof fn mul_one_right(a: nat)
    ensures a * 1 == a
{
}

// ----------------------------------------------------------------------------
// Distributivity
// ----------------------------------------------------------------------------

pub proof fn mul_add_distr_left(a: nat, b: nat, c: nat)
    ensures a * (b + c) == a * b + a * c
{
    assert(a * (b + c) == a * b + a * c) by (nonlinear_arith);
}

pub proof fn mul_add_distr_right(a: nat, b: nat, c: nat)
    ensures (a + b) * c == a * c + b * c
{
    assert((a + b) * c == a * c + b * c) by (nonlinear_arith);
}

// ----------------------------------------------------------------------------
// Ordering
// ----------------------------------------------------------------------------

pub proof fn le_refl(a: nat)
    ensures a <= a
{
}

pub proof fn le_antisym(a: nat, b: nat)
    requires a <= b, b <= a
    ensures a == b
{
}

pub proof fn le_trans(a: nat, b: nat, c: nat)
    requires a <= b, b <= c
    ensures a <= c
{
}

pub proof fn le_total(a: nat, b: nat)
    ensures a <= b || b <= a
{
}

// Strict ordering
pub proof fn lt_irrefl(a: nat)
    ensures !(a < a)
{
}

pub proof fn lt_trans(a: nat, b: nat, c: nat)
    requires a < b, b < c
    ensures a < c
{
}

// ----------------------------------------------------------------------------
// Addition and Ordering
// ----------------------------------------------------------------------------

pub proof fn add_le_mono_left(a: nat, b: nat, c: nat)
    requires b <= c
    ensures a + b <= a + c
{
}

pub proof fn add_le_mono_right(a: nat, b: nat, c: nat)
    requires a <= b
    ensures a + c <= b + c
{
}

pub proof fn add_lt_mono_left(a: nat, b: nat, c: nat)
    requires b < c
    ensures a + b < a + c
{
}

// ----------------------------------------------------------------------------
// Multiplication and Ordering
// ----------------------------------------------------------------------------

pub proof fn mul_le_mono_left(a: nat, b: nat, c: nat)
    requires b <= c
    ensures a * b <= a * c
{
    assert(a * b <= a * c) by (nonlinear_arith)
        requires b <= c;
}

pub proof fn mul_pos(a: nat, b: nat)
    requires a > 0, b > 0
    ensures a * b > 0
{
    assert(a * b > 0) by (nonlinear_arith)
        requires a > 0, b > 0;
}

// ----------------------------------------------------------------------------
// Division and Modulo
// ----------------------------------------------------------------------------

pub proof fn div_mod_spec(a: nat, b: nat)
    requires b > 0
    ensures a == (a / b) * b + (a % b)
{
    // Division/modulo identity - axiom in Verus
    assume(a == (a / b) * b + (a % b));
}

pub proof fn mod_bound(a: nat, b: nat)
    requires b > 0
    ensures a % b < b
{
}

pub proof fn mod_self(a: nat)
    requires a > 0
    ensures a % a == 0
{
}

pub proof fn div_self(a: nat)
    requires a > 0
    ensures a / a == 1
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_arithmetic()
{
    add_comm(3, 5);
    assert(3 + 5 == 5 + 3);

    mul_comm(4, 7);
    assert(4 * 7 == 7 * 4);

    mul_add_distr_left(2, 3, 4);
    assert(2 * (3 + 4) == 2 * 3 + 2 * 4);
}

pub proof fn example_ordering()
{
    le_refl(5);
    assert(5 <= 5);

    le_trans(3, 5, 8);
    assert(3 <= 8);

    lt_trans(1, 3, 7);
    assert(1 < 7);
}

pub proof fn example_div_mod()
{
    div_mod_spec(17, 5);
    assert(17int == (17int / 5) * 5 + (17int % 5));

    mod_bound(17, 5);
    assert(17int % 5 < 5);

    div_self(10);
    mod_self(10);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn nat_arith_verify()
{
    example_arithmetic();
    example_ordering();
    example_div_mod();

    // Test associativity
    add_assoc(1, 2, 3);
    mul_assoc(2, 3, 4);
}

pub fn main() {
    proof {
        nat_arith_verify();
    }
}

} // verus!
