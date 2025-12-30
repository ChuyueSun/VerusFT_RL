use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Comparison Operations (vfa-current/Perm, Sort)
// Comparison functions and their properties
// ============================================================================

// ----------------------------------------------------------------------------
// Basic Comparisons
// ----------------------------------------------------------------------------

// Less than or equal
pub open spec fn leb(a: nat, b: nat) -> bool {
    a <= b
}

// Less than
pub open spec fn ltb(a: nat, b: nat) -> bool {
    a < b
}

// Greater than or equal
pub open spec fn geb(a: nat, b: nat) -> bool {
    a >= b
}

// Greater than
pub open spec fn gtb(a: nat, b: nat) -> bool {
    a > b
}

// Equality
pub open spec fn eqb(a: nat, b: nat) -> bool {
    a == b
}

// ----------------------------------------------------------------------------
// Comparison Properties
// ----------------------------------------------------------------------------

// Reflexivity of <=
pub proof fn leb_refl(a: nat)
    ensures leb(a, a)
{
}

// Antisymmetry of <=
pub proof fn leb_antisym(a: nat, b: nat)
    requires leb(a, b), leb(b, a)
    ensures a == b
{
}

// Transitivity of <=
pub proof fn leb_trans(a: nat, b: nat, c: nat)
    requires leb(a, b), leb(b, c)
    ensures leb(a, c)
{
}

// Totality of <=
pub proof fn leb_total(a: nat, b: nat)
    ensures leb(a, b) || leb(b, a)
{
}

// Strict order is irreflexive
pub proof fn ltb_irrefl(a: nat)
    ensures !ltb(a, a)
{
}

// Strict order is transitive
pub proof fn ltb_trans(a: nat, b: nat, c: nat)
    requires ltb(a, b), ltb(b, c)
    ensures ltb(a, c)
{
}

// ----------------------------------------------------------------------------
// Relationships Between Comparisons
// ----------------------------------------------------------------------------

// < implies <=
pub proof fn ltb_implies_leb(a: nat, b: nat)
    requires ltb(a, b)
    ensures leb(a, b)
{
}

// <= and != implies <
pub proof fn leb_neq_ltb(a: nat, b: nat)
    requires leb(a, b), a != b
    ensures ltb(a, b)
{
}

// Not <= implies >
pub proof fn not_leb_gtb(a: nat, b: nat)
    requires !leb(a, b)
    ensures gtb(a, b)
{
}

// Not < implies >=
pub proof fn not_ltb_geb(a: nat, b: nat)
    requires !ltb(a, b)
    ensures geb(a, b)
{
}

// Complement relationships
pub proof fn leb_gtb_false(a: nat, b: nat)
    requires leb(a, b)
    ensures !gtb(a, b)
{
}

pub proof fn ltb_geb_false(a: nat, b: nat)
    requires ltb(a, b)
    ensures !geb(a, b)
{
}

// ----------------------------------------------------------------------------
// Trichotomy
// ----------------------------------------------------------------------------

// Exactly one of <, =, > holds
pub proof fn trichotomy(a: nat, b: nat)
    ensures (ltb(a, b) && !eqb(a, b) && !gtb(a, b)) ||
            (!ltb(a, b) && eqb(a, b) && !gtb(a, b)) ||
            (!ltb(a, b) && !eqb(a, b) && gtb(a, b))
{
}

// ----------------------------------------------------------------------------
// Min and Max
// ----------------------------------------------------------------------------

pub open spec fn min(a: nat, b: nat) -> nat {
    if a <= b { a } else { b }
}

pub open spec fn max(a: nat, b: nat) -> nat {
    if a >= b { a } else { b }
}

// Min properties
pub proof fn min_comm(a: nat, b: nat)
    ensures min(a, b) == min(b, a)
{
}

pub proof fn min_assoc(a: nat, b: nat, c: nat)
    ensures min(min(a, b), c) == min(a, min(b, c))
{
}

pub proof fn min_le_left(a: nat, b: nat)
    ensures min(a, b) <= a
{
}

pub proof fn min_le_right(a: nat, b: nat)
    ensures min(a, b) <= b
{
}

// Max properties
pub proof fn max_comm(a: nat, b: nat)
    ensures max(a, b) == max(b, a)
{
}

pub proof fn max_assoc(a: nat, b: nat, c: nat)
    ensures max(max(a, b), c) == max(a, max(b, c))
{
}

pub proof fn max_ge_left(a: nat, b: nat)
    ensures max(a, b) >= a
{
}

pub proof fn max_ge_right(a: nat, b: nat)
    ensures max(a, b) >= b
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_comparisons()
{
    assert(leb(3, 5));
    assert(leb(5, 5));
    assert(!leb(7, 5));

    assert(ltb(3, 5));
    assert(!ltb(5, 5));

    assert(geb(5, 3));
    assert(geb(5, 5));

    assert(gtb(7, 5));
    assert(!gtb(5, 5));
}

pub proof fn example_min_max()
{
    assert(min(3, 7) == 3);
    assert(min(7, 3) == 3);
    assert(min(5, 5) == 5);

    assert(max(3, 7) == 7);
    assert(max(7, 3) == 7);
    assert(max(5, 5) == 5);
}

pub proof fn example_trichotomy()
{
    // 3 < 5
    trichotomy(3, 5);
    assert(ltb(3, 5));

    // 5 = 5
    trichotomy(5, 5);
    assert(eqb(5, 5));

    // 7 > 5
    trichotomy(7, 5);
    assert(gtb(7, 5));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn comparison_verify()
{
    example_comparisons();
    example_min_max();
    example_trichotomy();

    // Test transitivity
    leb_trans(2, 5, 10);
    assert(leb(2, 10));

    ltb_trans(1, 3, 7);
    assert(ltb(1, 7));

    // Test min/max properties
    min_comm(10, 20);
    max_comm(10, 20);
    min_assoc(5, 10, 15);
    max_assoc(5, 10, 15);
}

pub fn main() {
    proof {
        comparison_verify();
    }
}

} // verus!
