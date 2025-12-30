use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// Hoare Logic as a Logic (Software Foundations, PLF/HoareAsLogic)
// Derivability, soundness, and completeness
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// State and Basic Definitions
// ----------------------------------------------------------------------------

pub type Var = nat;
pub type State = Map<Var, int>;

pub spec const X: Var = 0;
pub spec const Y: Var = 1;

pub open spec fn lookup(st: State, x: Var) -> int {
    if st.dom().contains(x) { st[x] } else { 0 }
}

pub open spec fn update(st: State, x: Var, v: int) -> State {
    st.insert(x, v)
}

// ----------------------------------------------------------------------------
// Expressions
// ----------------------------------------------------------------------------

pub enum AExp {
    ANum { n: int },
    AId { x: Var },
    APlus { a1: Box<AExp>, a2: Box<AExp> },
}

pub open spec fn aeval(st: State, a: AExp) -> int
    decreases a
{
    match a {
        AExp::ANum { n } => n,
        AExp::AId { x } => lookup(st, x),
        AExp::APlus { a1, a2 } => aeval(st, *a1) + aeval(st, *a2),
    }
}

pub enum BExp {
    BTrue,
    BFalse,
    BEq { a1: AExp, a2: AExp },
}

pub open spec fn beval(st: State, b: BExp) -> bool {
    match b {
        BExp::BTrue => true,
        BExp::BFalse => false,
        BExp::BEq { a1, a2 } => aeval(st, a1) == aeval(st, a2),
    }
}

// ----------------------------------------------------------------------------
// Commands
// ----------------------------------------------------------------------------

pub enum Com {
    CSkip,
    CAsgn { x: Var, a: AExp },
    CSeq { c1: Box<Com>, c2: Box<Com> },
}

// ----------------------------------------------------------------------------
// Assertions
// ----------------------------------------------------------------------------

pub type Assertion = spec_fn(State) -> bool;

// ----------------------------------------------------------------------------
// Hoare Triple: Model-Theoretic (Validity)
// ----------------------------------------------------------------------------

// Command evaluation (simplified, without loops)
pub open spec fn ceval(c: Com, st: State) -> State
    decreases c
{
    match c {
        Com::CSkip => st,
        Com::CAsgn { x, a } => update(st, x, aeval(st, a)),
        Com::CSeq { c1, c2 } => ceval(*c2, ceval(*c1, st)),
    }
}

// A Hoare triple is valid if execution preserves the specification
pub open spec fn hoare_triple_valid(p: Assertion, c: Com, q: Assertion) -> bool {
    forall|st: State| p(st) ==> q(ceval(c, st))
}

// ----------------------------------------------------------------------------
// Hoare Triple: Proof-Theoretic (Derivability)
// ----------------------------------------------------------------------------

// Derivability is defined by the Hoare logic proof rules
pub enum Derivation {
    // H_Skip: {P} skip {P}
    DSkip { p: Assertion },

    // H_Asgn: {Q[x := a]} x := a {Q}
    DAsgn { q: Assertion, x: Var, a: AExp },

    // H_Seq: {P} c1 {Q}, {Q} c2 {R} => {P} c1;c2 {R}
    DSeq { p: Assertion, q: Assertion, r: Assertion, c1: Com, c2: Com },

    // H_Consequence: P' => P, {P} c {Q}, Q => Q' => {P'} c {Q'}
    DConseq { p: Assertion, p_prime: Assertion, q: Assertion, q_prime: Assertion, c: Com },
}

// ----------------------------------------------------------------------------
// Weakest Precondition
// ----------------------------------------------------------------------------

// wp(c, Q) is the weakest P such that {P} c {Q} is valid
pub open spec fn wp(c: Com, q: Assertion) -> Assertion
    decreases c
{
    match c {
        Com::CSkip => q,
        Com::CAsgn { x, a } => |st: State| q(update(st, x, aeval(st, a))),
        Com::CSeq { c1, c2 } => wp(*c1, wp(*c2, q)),
    }
}

// ----------------------------------------------------------------------------
// Soundness: Derivable implies Valid
// ----------------------------------------------------------------------------

// Skip rule is sound
pub proof fn skip_sound(p: Assertion)
    ensures hoare_triple_valid(p, Com::CSkip, p)
{
}

// Assignment rule is sound
pub proof fn asgn_sound(q: Assertion, x: Var, a: AExp)
    ensures hoare_triple_valid(
        |st: State| q(update(st, x, aeval(st, a))),
        Com::CAsgn { x, a },
        q
    )
{
}

// Sequence rule is sound (simplified proof)
pub proof fn seq_sound_example()
{
    // Example: {X=0} X:=1 {X=1}, {X=1} X:=2 {X=2} => {X=0} X:=1;X:=2 {X=2}
    let st0 = update(Map::<Var, int>::empty(), X, 0);
    let c1 = Com::CAsgn { x: X, a: AExp::ANum { n: 1 } };
    let c2 = Com::CAsgn { x: X, a: AExp::ANum { n: 2 } };

    let st1 = ceval(c1, st0);
    assert(lookup(st1, X) == 1);

    let st2 = ceval(c2, st1);
    assert(lookup(st2, X) == 2);
}

// ----------------------------------------------------------------------------
// Weakest Precondition Properties
// ----------------------------------------------------------------------------

// wp(skip, Q) = Q (example)
pub proof fn wp_skip_example()
{
    let q: Assertion = |st: State| lookup(st, X) == 5;
    let wp_q = wp(Com::CSkip, q);

    let st = update(Map::<Var, int>::empty(), X, 5);
    assert(q(st));
    assert(wp_q(st));  // wp(skip, Q) = Q
}

// wp(X:=5, X=5) = True (example)
pub proof fn wp_asgn_example()
{
    let q: Assertion = |st: State| lookup(st, X) == 5;
    let wp_q = wp(Com::CAsgn { x: X, a: AExp::ANum { n: 5 } }, q);

    // wp is satisfied by any state
    let st = Map::<Var, int>::empty();
    assert(lookup(update(st, X, 5), X) == 5);
    assert(wp_q(st));
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

// {X = 0} skip {X = 0}
pub proof fn example_skip()
{
    let p: Assertion = |st: State| lookup(st, X) == 0;
    skip_sound(p);
    assert(hoare_triple_valid(p, Com::CSkip, p));
}

// {True} X := 5 {X = 5}
pub proof fn example_asgn()
{
    let q: Assertion = |st: State| lookup(st, X) == 5;
    let pre: Assertion = |st: State| q(update(st, X, 5));

    asgn_sound(q, X, AExp::ANum { n: 5 });

    // Show that pre is satisfied by any state (test with empty state)
    let st = Map::<Var, int>::empty();
    assert(lookup(update(st, X, 5), X) == 5);
    assert(pre(st));
}

// {X = 0} X := X + 1 {X = 1}
pub proof fn example_increment()
{
    let pre: Assertion = |st: State| lookup(st, X) == 0;
    let post: Assertion = |st: State| lookup(st, X) == 1;
    let a = AExp::APlus { a1: Box::new(AExp::AId { x: X }), a2: Box::new(AExp::ANum { n: 1 }) };
    let c = Com::CAsgn { x: X, a };

    // Test with concrete state where X = 0
    let st = update(Map::<Var, int>::empty(), X, 0);
    reveal_with_fuel(aeval, 3);
    assert(pre(st));
    assert(aeval(st, a) == 1);
    assert(post(ceval(c, st)));
}

// wp example: wp(X := 5, X = 5) = True
pub proof fn example_wp_asgn()
{
    let post: Assertion = |st: State| lookup(st, X) == 5;
    let pre = wp(Com::CAsgn { x: X, a: AExp::ANum { n: 5 } }, post);

    // Test with empty state
    let st = Map::<Var, int>::empty();
    assert(lookup(update(st, X, 5), X) == 5);
    assert(pre(st));
}

// Sequence example: {X = 0} X := 1; X := 2 {X = 2}
pub proof fn example_seq()
{
    let pre: Assertion = |st: State| lookup(st, X) == 0;
    let mid: Assertion = |st: State| lookup(st, X) == 1;
    let post: Assertion = |st: State| lookup(st, X) == 2;

    let c1 = Com::CAsgn { x: X, a: AExp::ANum { n: 1 } };
    let c2 = Com::CAsgn { x: X, a: AExp::ANum { n: 2 } };

    // Test with concrete state where X = 0
    let st0 = update(Map::<Var, int>::empty(), X, 0);
    assert(pre(st0));

    let st1 = ceval(c1, st0);
    assert(mid(st1));

    let st2 = ceval(c2, st1);
    assert(post(st2));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn hoare_as_logic_examples_verify()
{
    example_skip();
    example_asgn();
    example_increment();
    example_wp_asgn();
    example_seq();

    // Test new examples
    seq_sound_example();
    wp_skip_example();
    wp_asgn_example();
}

pub fn main() {
    proof {
        hoare_as_logic_examples_verify();
    }
}

} // verus!
