use vstd::prelude::*;

verus! {

// AltAuto (Software Foundations, LF/AltAuto) style snippets in Verus.
//
// Focus: when automation gets stuck, restructure the statement so the SMT solver
// has good instantiation triggers (which must be *function calls* in Verus).

// A small trigger-friendly equality predicate for nat.
pub open spec fn eq_nat(x: nat, y: nat) -> bool { x == y }

// Uninterpreted functions, used to demonstrate trigger-driven instantiation.
pub uninterp spec fn f(x: int) -> int;
pub uninterp spec fn g(x: int) -> int;

// If we want the solver to instantiate on occurrences of `f(t)`, we make that
// the trigger explicitly.
pub broadcast axiom fn axiom_fg(x: int)
    ensures #[trigger] f(x) == g(x)
;

// Another common pattern: encode a predicate as a function call.
pub uninterp spec fn p(x: int) -> bool;

pub broadcast axiom fn axiom_p_implies_p(x: int)
    ensures #[trigger] p(x) ==> p(x)
;

// ----------------------------
// Triggered forall/exists examples
// ----------------------------

pub proof fn lemma_forall_with_explicit_trigger()
    ensures forall|x: int| #[trigger] f(x) == f(x)
{
    // Nothing to prove: it's reflexivity, but the trigger annotation makes the
    // quantifier well-formed and usable.
}

pub proof fn lemma_axiom_instantiation_1()
    ensures f(7) == g(7)
{
    axiom_fg(7);
    assert(f(7) == g(7));
}

pub proof fn lemma_axiom_instantiation_2(t: int)
    ensures f(t) == g(t)
{
    axiom_fg(t);
    assert(f(t) == g(t));
}

pub proof fn lemma_forall_p_is_trivial()
    ensures forall|x: int| #[trigger] p(x) ==> p(x)
{
    // This follows directly from the broadcast axiom (triggered on p(x)).
    assert(forall|x: int| #[trigger] p(x) ==> p(x));
}

pub proof fn lemma_exists_with_trigger(n: nat)
    ensures exists|m: nat| #[trigger] eq_nat(m, n)
{
    assert(exists|m: nat| #[trigger] eq_nat(m, n)) by {
        let m = n;
        assert(eq_nat(m, n));
    }
}

pub proof fn lemma_choose_with_trigger(n: nat)
    ensures ({
        let m = choose|m: nat| eq_nat(m, n);
        eq_nat(m, n)
    })
{
    lemma_exists_with_trigger(n);
    let m = choose|m: nat| eq_nat(m, n);
    assert(eq_nat(m, n));
}

// ----------------------------
// Alternative automation: replace complex goals with triggerable helper calls
// ----------------------------

pub open spec fn twice(x: int) -> int { x + x }

pub proof fn lemma_twice_commutes(a: int, b: int)
    ensures twice(a + b) == twice(b + a)
{
    // Often, introducing helper functions (like twice) gives the solver
    // a stable function-call term to match on.
    assert(a + b == b + a);
}

pub fn main() {
    proof {
        lemma_forall_with_explicit_trigger();
        lemma_axiom_instantiation_1();
        lemma_axiom_instantiation_2(-3);
        lemma_forall_p_is_trivial();
        lemma_exists_with_trigger(5);
        lemma_choose_with_trigger(9);
        lemma_twice_commutes(10, 20);
    }
}

} // verus!
