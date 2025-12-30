use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Propositional Logic (Supporting VFA)
// Propositional connectives and tautologies
// ============================================================================

// ----------------------------------------------------------------------------
// Tautologies
// ----------------------------------------------------------------------------

// Law of excluded middle
pub proof fn excluded_middle(p: bool)
    ensures p || !p
{
}

// Non-contradiction
pub proof fn non_contradiction(p: bool)
    ensures !(p && !p)
{
}

// Double negation elimination
pub proof fn dne(p: bool)
    ensures !!p == p
{
}

// ----------------------------------------------------------------------------
// Syllogisms
// ----------------------------------------------------------------------------

// Modus ponens: P, P -> Q |- Q
pub proof fn modus_ponens(p: bool, q: bool)
    requires p, p ==> q
    ensures q
{
}

// Modus tollens: ~Q, P -> Q |- ~P
pub proof fn modus_tollens(p: bool, q: bool)
    requires !q, p ==> q
    ensures !p
{
}

// Hypothetical syllogism: P -> Q, Q -> R |- P -> R
pub proof fn hypothetical_syllogism(p: bool, q: bool, r: bool)
    requires p ==> q, q ==> r
    ensures p ==> r
{
}

// Disjunctive syllogism: P v Q, ~P |- Q
pub proof fn disjunctive_syllogism(p: bool, q: bool)
    requires p || q, !p
    ensures q
{
}

// ----------------------------------------------------------------------------
// Conjunction Rules
// ----------------------------------------------------------------------------

pub proof fn and_intro(p: bool, q: bool)
    requires p, q
    ensures p && q
{
}

pub proof fn and_elim_left(p: bool, q: bool)
    requires p && q
    ensures p
{
}

pub proof fn and_elim_right(p: bool, q: bool)
    requires p && q
    ensures q
{
}

// ----------------------------------------------------------------------------
// Disjunction Rules
// ----------------------------------------------------------------------------

pub proof fn or_intro_left(p: bool, q: bool)
    requires p
    ensures p || q
{
}

pub proof fn or_intro_right(p: bool, q: bool)
    requires q
    ensures p || q
{
}

pub proof fn or_elim(p: bool, q: bool, r: bool)
    requires p || q, p ==> r, q ==> r
    ensures r
{
}

// ----------------------------------------------------------------------------
// Implication Rules
// ----------------------------------------------------------------------------

pub proof fn implies_intro(p: bool, q: bool)
    requires p ==> q
    ensures p ==> q
{
}

pub proof fn implies_trans(p: bool, q: bool, r: bool)
    requires p ==> q, q ==> r
    ensures p ==> r
{
}

// ----------------------------------------------------------------------------
// Equivalence Rules
// ----------------------------------------------------------------------------

pub proof fn iff_intro(p: bool, q: bool)
    requires p ==> q, q ==> p
    ensures p <==> q
{
}

pub proof fn iff_elim_left(p: bool, q: bool)
    requires p <==> q, p
    ensures q
{
}

pub proof fn iff_elim_right(p: bool, q: bool)
    requires p <==> q, q
    ensures p
{
}

// ----------------------------------------------------------------------------
// Negation Rules
// ----------------------------------------------------------------------------

// Proof by contradiction
pub proof fn contradiction_elim(p: bool)
    requires false
    ensures p
{
}

// Negation introduction
pub proof fn neg_intro(p: bool)
    requires p ==> false
    ensures !p
{
}

// ----------------------------------------------------------------------------
// Classical Logic
// ----------------------------------------------------------------------------

// Peirce's law
pub proof fn peirce(p: bool, q: bool)
    ensures ((p ==> q) ==> p) ==> p
{
}

// Contraposition
pub proof fn contraposition(p: bool, q: bool)
    ensures (p ==> q) <==> (!q ==> !p)
{
}

// Material implication
pub proof fn material_impl(p: bool, q: bool)
    ensures (p ==> q) <==> (!p || q)
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_syllogism()
{
    // If it rains, the ground is wet
    // It rains
    // Therefore, the ground is wet
    let rains = true;
    let ground_wet = true;
    modus_ponens(rains, ground_wet);
}

pub proof fn example_case_analysis()
{
    // Either P or Q
    // If P then R
    // If Q then R
    // Therefore R
    let p = true;
    let q = false;
    let r = true;

    or_elim(p, q, r);
}

pub proof fn example_chain()
{
    // P -> Q
    // Q -> R
    // R -> S
    // Therefore P -> S
    let p = true;
    let q = true;
    let r = true;
    let s = true;

    hypothetical_syllogism(p, q, r);
    hypothetical_syllogism(p, r, s);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn logic_prop_verify()
{
    // Test tautologies
    excluded_middle(true);
    excluded_middle(false);
    non_contradiction(true);
    dne(true);
    dne(false);

    // Test syllogisms
    example_syllogism();
    example_case_analysis();
    example_chain();

    // Test classical logic
    peirce(true, false);
    contraposition(true, false);
    material_impl(true, true);
}

pub fn main() {
    proof {
        logic_prop_verify();
    }
}

} // verus!
