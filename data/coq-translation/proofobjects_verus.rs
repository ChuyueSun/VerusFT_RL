use vstd::prelude::*;

verus! {

// ProofObjects (Software Foundations, LF/ProofObjects) style snippets in Verus.
//
// Theme: proofs are explicit objects (values) that can be passed around.
// In Verus, this corresponds to `proof fn` values/arguments plus ghost/spec reasoning.

// ----------------------------
// Propositions as booleans
// ----------------------------

pub proof fn lemma_true()
    ensures true
{
}

pub proof fn lemma_and_intro(a: bool, b: bool)
    requires a,
        b,
    ensures a && b
{
}

pub proof fn lemma_and_elim_left(a: bool, b: bool)
    requires a && b,
    ensures a
{
}

pub proof fn lemma_and_elim_right(a: bool, b: bool)
    requires a && b,
    ensures b
{
}

pub proof fn lemma_or_intro_left(a: bool, b: bool)
    requires a,
    ensures a || b
{
}

pub proof fn lemma_or_intro_right(a: bool, b: bool)
    requires b,
    ensures a || b
{
}

pub proof fn lemma_imp_trans(a: bool, b: bool, c: bool)
    requires a ==> b,
        b ==> c,
    ensures a ==> c
{
    // Proof object: a function from a proof of a to a proof of c.
    assert(a ==> c) by {
        if a {
            assert(b);
            assert(c);
        }
    }
}

pub proof fn lemma_contra(a: bool, b: bool)
    requires a ==> b,
        a ==> !b,
    ensures !a
{
    if a {
        assert(b);
        assert(!b);
    }
}

// ----------------------------
// Small proof-carrying "evidence" tokens
// ----------------------------

// We use private fields + constructors to make these feel like proof objects.
// (This is only a didactic pattern; Verus doesn't enforce privacy against this module.)

pub open spec fn is_even(n: nat) -> bool {
    n % 2 == 0
}

pub open spec fn is_mul4(n: nat) -> bool {
    n % 4 == 0
}

pub struct EvenEv {
    pub n: nat,
}

pub struct Mul4Ev {
    pub n: nat,
}

pub proof fn even_ev_new(n: nat) -> (ev: EvenEv)
    requires is_even(n)
    ensures ev.n == n,
        is_even(ev.n)
{
    EvenEv { n }
}

pub proof fn mul4_ev_new(n: nat) -> (ev: Mul4Ev)
    requires is_mul4(n)
    ensures ev.n == n,
        is_mul4(ev.n)
{
    Mul4Ev { n }
}

pub proof fn mul4_implies_even(ev4: Mul4Ev) -> (ev2: EvenEv)
    requires is_mul4(ev4.n)
    ensures ev2.n == ev4.n,
        is_even(ev2.n)
{
    // Arithmetic fact: 4|n => 2|n
    assert(is_even(ev4.n)) by {
        // If n % 4 == 0 then n % 2 == 0
        // Verus can discharge this via nonlinear arithmetic on mod.
    }
    even_ev_new(ev4.n)
}

// ----------------------------
// Proof objects for quantifiers
// ----------------------------

pub open spec fn add0(x: nat) -> nat { x + 0 }
pub open spec fn id_nat(x: nat) -> nat { x }
pub open spec fn eq_nat(x: nat, y: nat) -> bool { x == y }

pub proof fn lemma_forall_instantiation()
{
    // Proving universal facts directly as proof objects.
    assert(forall|x: nat| #[trigger] add0(x) == x);
    assert(forall|x: nat| #[trigger] id_nat(x) == x);

    // Instantiating a universal fact at a particular point.
    let y: nat = 5;
    assert(y + 0 == y);
}

pub proof fn lemma_exists_witness(n: nat)
    ensures exists|m: nat| #[trigger] eq_nat(m, n)
{
    assert(exists|m: nat| #[trigger] eq_nat(m, n)) by {
        let m = n;
        assert(eq_nat(m, n));
    }
}

pub proof fn lemma_choose_example(n: nat)
    ensures ({
        let m = choose|m: nat| eq_nat(m, n);
        eq_nat(m, n)
    })
{
    lemma_exists_witness(n);
    let m = choose|m: nat| eq_nat(m, n);
    assert(eq_nat(m, n));
}

// ----------------------------
// A tiny "proof term" pipeline
// ----------------------------

pub proof fn lemma_pipeline_example(a: bool, b: bool, c: bool)
    requires a,
        a ==> b,
        b ==> c,
    ensures c
{
    // Think of `a` as a proof term for proposition `a`.
    assert(b);
    assert(c);
}

pub fn main() {
    proof {
        lemma_true();
        lemma_and_intro(true, true);
        lemma_and_elim_left(true, true);
        lemma_and_elim_right(true, true);
        lemma_or_intro_left(true, false);
        lemma_or_intro_right(false, true);

        lemma_imp_trans(true, true, true);
        lemma_contra(false, true);

        // Evidence objects
        let ev4 = mul4_ev_new(8);
        let _ev2 = mul4_implies_even(ev4);

        lemma_forall_instantiation();
        lemma_exists_witness(5);
        lemma_choose_example(7);
        lemma_pipeline_example(true, true, true);
    }
}

} // verus!
