use vstd::prelude::*;
use vstd::map::*;
use vstd::set::*;

verus! {

// ============================================================================
// Equiv (Software Foundations, PLF/Equiv) in Verus
// Program Equivalence for the IMP language
// ============================================================================

broadcast use { group_map_axioms, group_set_axioms };

// ----------------------------------------------------------------------------
// State and Variables
// ----------------------------------------------------------------------------

pub type Var = nat;
pub type Store = Map<Var, int>;

pub open spec fn store_empty() -> Store {
    Map::<Var, int>::empty()
}

pub open spec fn store_update(st: Store, x: Var, v: int) -> Store {
    st.insert(x, v)
}

pub open spec fn store_get(st: Store, x: Var) -> int {
    if st.dom().contains(x) { st[x] } else { 0 }
}

// Variable names for examples
pub spec const X: Var = 0;
pub spec const Y: Var = 1;
pub spec const Z: Var = 2;

// ----------------------------------------------------------------------------
// Arithmetic Expressions
// ----------------------------------------------------------------------------

pub enum AExp {
    ANum { n: int },
    AId { x: Var },
    APlus { a1: Box<AExp>, a2: Box<AExp> },
    AMinus { a1: Box<AExp>, a2: Box<AExp> },
    AMult { a1: Box<AExp>, a2: Box<AExp> },
}

pub open spec fn aeval(a: AExp, st: Store) -> int
    decreases a
{
    match a {
        AExp::ANum { n } => n,
        AExp::AId { x } => store_get(st, x),
        AExp::APlus { a1, a2 } => aeval(*a1, st) + aeval(*a2, st),
        AExp::AMinus { a1, a2 } => aeval(*a1, st) - aeval(*a2, st),
        AExp::AMult { a1, a2 } => aeval(*a1, st) * aeval(*a2, st),
    }
}

// ----------------------------------------------------------------------------
// Boolean Expressions
// ----------------------------------------------------------------------------

pub enum BExp {
    BTrue,
    BFalse,
    BEq { a1: Box<AExp>, a2: Box<AExp> },
    BLe { a1: Box<AExp>, a2: Box<AExp> },
    BNot { b1: Box<BExp> },
    BAnd { b1: Box<BExp>, b2: Box<BExp> },
}

pub open spec fn beval(b: BExp, st: Store) -> bool
    decreases b
{
    match b {
        BExp::BTrue => true,
        BExp::BFalse => false,
        BExp::BEq { a1, a2 } => aeval(*a1, st) == aeval(*a2, st),
        BExp::BLe { a1, a2 } => aeval(*a1, st) <= aeval(*a2, st),
        BExp::BNot { b1 } => !beval(*b1, st),
        BExp::BAnd { b1, b2 } => beval(*b1, st) && beval(*b2, st),
    }
}

// ----------------------------------------------------------------------------
// Commands
// ----------------------------------------------------------------------------

pub enum Com {
    CSkip,
    CAsgn { x: Var, a: Box<AExp> },
    CSeq { c1: Box<Com>, c2: Box<Com> },
    CIf { b: Box<BExp>, ct: Box<Com>, cf: Box<Com> },
    CWhile { b: Box<BExp>, body: Box<Com> },
}

// Command evaluation with fuel (to handle potential non-termination)
pub open spec fn ceval(fuel: nat, c: Com, st: Store) -> Option<Store>
    decreases fuel, c
{
    if fuel == 0 {
        Option::<Store>::None
    } else {
        let fuel1 = (fuel - 1) as nat;
        match c {
            Com::CSkip => Option::Some(st),
            Com::CAsgn { x, a } => Option::Some(store_update(st, x, aeval(*a, st))),
            Com::CSeq { c1, c2 } => {
                match ceval(fuel1, *c1, st) {
                    Option::None => Option::<Store>::None,
                    Option::Some(st1) => ceval(fuel1, *c2, st1),
                }
            }
            Com::CIf { b, ct, cf } => {
                if beval(*b, st) {
                    ceval(fuel1, *ct, st)
                } else {
                    ceval(fuel1, *cf, st)
                }
            }
            Com::CWhile { b, body } => {
                if beval(*b, st) {
                    match ceval(fuel1, *body, st) {
                        Option::None => Option::<Store>::None,
                        Option::Some(st1) => ceval(fuel1, Com::CWhile { b: Box::new(*b), body: Box::new(*body) }, st1),
                    }
                } else {
                    Option::Some(st)
                }
            }
        }
    }
}

// ============================================================================
// BEHAVIORAL EQUIVALENCE DEFINITIONS
// ============================================================================

// Two arithmetic expressions are equivalent if they evaluate to the same
// result in every state.
pub open spec fn aequiv(a1: AExp, a2: AExp) -> bool {
    forall|st: Store| aeval(a1, st) == aeval(a2, st)
}

// Two boolean expressions are equivalent if they evaluate to the same
// result in every state.
pub open spec fn bequiv(b1: BExp, b2: BExp) -> bool {
    forall|st: Store| beval(b1, st) == beval(b2, st)
}

// Two commands are behaviorally equivalent if, for any given starting state,
// they either both diverge or terminate in the same final state.
// (Using fuel-based semantics: equivalent for all fuel levels)
pub open spec fn cequiv(c1: Com, c2: Com) -> bool {
    forall|fuel: nat, st: Store| ceval(fuel, c1, st) == ceval(fuel, c2, st)
}

// A weaker notion: commands produce the same result when they terminate
pub open spec fn cequiv_terminating(c1: Com, c2: Com) -> bool {
    forall|fuel: nat, st: Store, st_final: Store|
        (ceval(fuel, c1, st) == Option::Some(st_final)) <==>
        (ceval(fuel, c2, st) == Option::Some(st_final))
}

// ============================================================================
// EQUIVALENCE PROPERTIES: REFLEXIVITY, SYMMETRY, TRANSITIVITY
// ============================================================================

// ----------------------------------------------------------------------------
// Arithmetic Expression Equivalence Properties
// ----------------------------------------------------------------------------

pub proof fn aequiv_refl(a: AExp)
    ensures aequiv(a, a)
{
    // Trivially true by definition
}

pub proof fn aequiv_sym(a1: AExp, a2: AExp)
    requires aequiv(a1, a2)
    ensures aequiv(a2, a1)
{
    // Follows from symmetry of equality
}

pub proof fn aequiv_trans(a1: AExp, a2: AExp, a3: AExp)
    requires
        aequiv(a1, a2),
        aequiv(a2, a3),
    ensures aequiv(a1, a3)
{
    // Follows from transitivity of equality
}

// ----------------------------------------------------------------------------
// Boolean Expression Equivalence Properties
// ----------------------------------------------------------------------------

pub proof fn bequiv_refl(b: BExp)
    ensures bequiv(b, b)
{
}

pub proof fn bequiv_sym(b1: BExp, b2: BExp)
    requires bequiv(b1, b2)
    ensures bequiv(b2, b1)
{
}

pub proof fn bequiv_trans(b1: BExp, b2: BExp, b3: BExp)
    requires
        bequiv(b1, b2),
        bequiv(b2, b3),
    ensures bequiv(b1, b3)
{
}

// ----------------------------------------------------------------------------
// Command Equivalence Properties
// ----------------------------------------------------------------------------

pub proof fn cequiv_refl(c: Com)
    ensures cequiv(c, c)
{
}

pub proof fn cequiv_sym(c1: Com, c2: Com)
    requires cequiv(c1, c2)
    ensures cequiv(c2, c1)
{
}

pub proof fn cequiv_trans(c1: Com, c2: Com, c3: Com)
    requires
        cequiv(c1, c2),
        cequiv(c2, c3),
    ensures cequiv(c1, c3)
{
}

// ============================================================================
// SIMPLE BEHAVIORAL EQUIVALENCE EXAMPLES
// ============================================================================

// skip; c is equivalent to c (for fuel >= 2)
// Note: With fuel-based semantics, we need sufficient fuel for both commands
pub proof fn skip_left_fuel2(c: Com, fuel: nat, st: Store)
    requires fuel >= 2
    ensures ceval(fuel, Com::CSeq { c1: Box::new(Com::CSkip), c2: Box::new(c) }, st)
        == ceval((fuel - 1) as nat, c, st)
{
    reveal_with_fuel(ceval, 2);
}

// if true then c1 else c2 evaluates to c1's result
pub proof fn if_true_fuel(c1: Com, c2: Com, fuel: nat, st: Store)
    requires fuel >= 2
    ensures ceval(fuel, Com::CIf { b: Box::new(BExp::BTrue), ct: Box::new(c1), cf: Box::new(c2) }, st)
        == ceval((fuel - 1) as nat, c1, st)
{
    reveal_with_fuel(ceval, 2);
}

// if false then c1 else c2 evaluates to c2's result
pub proof fn if_false_fuel(c1: Com, c2: Com, fuel: nat, st: Store)
    requires fuel >= 2
    ensures ceval(fuel, Com::CIf { b: Box::new(BExp::BFalse), ct: Box::new(c1), cf: Box::new(c2) }, st)
        == ceval((fuel - 1) as nat, c2, st)
{
    reveal_with_fuel(ceval, 2);
}

// while false do c end terminates immediately
pub proof fn while_false_fuel(c: Com, fuel: nat, st: Store)
    requires fuel >= 1
    ensures ceval(fuel, Com::CWhile { b: Box::new(BExp::BFalse), body: Box::new(c) }, st)
        == Option::Some(st)
{
    reveal_with_fuel(ceval, 2);
}

// skip terminates immediately
pub proof fn skip_terminates(fuel: nat, st: Store)
    requires fuel >= 1
    ensures ceval(fuel, Com::CSkip, st) == Option::Some(st)
{
    reveal_with_fuel(ceval, 1);
}

// ============================================================================
// CONSTANT FOLDING OPTIMIZATION
// ============================================================================

// Constant folding for arithmetic expressions
pub open spec fn fold_constants_aexp(a: AExp) -> AExp
    decreases a
{
    match a {
        AExp::ANum { n } => AExp::ANum { n },
        AExp::AId { x } => AExp::AId { x },
        AExp::APlus { a1, a2 } => {
            let a1_folded = fold_constants_aexp(*a1);
            let a2_folded = fold_constants_aexp(*a2);
            match (a1_folded, a2_folded) {
                (AExp::ANum { n: n1 }, AExp::ANum { n: n2 }) => AExp::ANum { n: n1 + n2 },
                (a1f, a2f) => AExp::APlus { a1: Box::new(a1f), a2: Box::new(a2f) },
            }
        }
        AExp::AMinus { a1, a2 } => {
            let a1_folded = fold_constants_aexp(*a1);
            let a2_folded = fold_constants_aexp(*a2);
            match (a1_folded, a2_folded) {
                (AExp::ANum { n: n1 }, AExp::ANum { n: n2 }) => AExp::ANum { n: n1 - n2 },
                (a1f, a2f) => AExp::AMinus { a1: Box::new(a1f), a2: Box::new(a2f) },
            }
        }
        AExp::AMult { a1, a2 } => {
            let a1_folded = fold_constants_aexp(*a1);
            let a2_folded = fold_constants_aexp(*a2);
            match (a1_folded, a2_folded) {
                (AExp::ANum { n: n1 }, AExp::ANum { n: n2 }) => AExp::ANum { n: n1 * n2 },
                (a1f, a2f) => AExp::AMult { a1: Box::new(a1f), a2: Box::new(a2f) },
            }
        }
    }
}

// Constant folding for boolean expressions
pub open spec fn fold_constants_bexp(b: BExp) -> BExp
    decreases b
{
    match b {
        BExp::BTrue => BExp::BTrue,
        BExp::BFalse => BExp::BFalse,
        BExp::BEq { a1, a2 } => {
            let a1_folded = fold_constants_aexp(*a1);
            let a2_folded = fold_constants_aexp(*a2);
            match (a1_folded, a2_folded) {
                (AExp::ANum { n: n1 }, AExp::ANum { n: n2 }) => {
                    if n1 == n2 { BExp::BTrue } else { BExp::BFalse }
                }
                (a1f, a2f) => BExp::BEq { a1: Box::new(a1f), a2: Box::new(a2f) },
            }
        }
        BExp::BLe { a1, a2 } => {
            let a1_folded = fold_constants_aexp(*a1);
            let a2_folded = fold_constants_aexp(*a2);
            match (a1_folded, a2_folded) {
                (AExp::ANum { n: n1 }, AExp::ANum { n: n2 }) => {
                    if n1 <= n2 { BExp::BTrue } else { BExp::BFalse }
                }
                (a1f, a2f) => BExp::BLe { a1: Box::new(a1f), a2: Box::new(a2f) },
            }
        }
        BExp::BNot { b1 } => {
            let b1_folded = fold_constants_bexp(*b1);
            match b1_folded {
                BExp::BTrue => BExp::BFalse,
                BExp::BFalse => BExp::BTrue,
                _ => BExp::BNot { b1: Box::new(b1_folded) },
            }
        }
        BExp::BAnd { b1, b2 } => {
            let b1_folded = fold_constants_bexp(*b1);
            let b2_folded = fold_constants_bexp(*b2);
            match (b1_folded, b2_folded) {
                (BExp::BTrue, b2f) => b2f,
                (BExp::BFalse, _) => BExp::BFalse,
                (b1f, BExp::BTrue) => b1f,
                (_, BExp::BFalse) => BExp::BFalse,
                (b1f, b2f) => BExp::BAnd { b1: Box::new(b1f), b2: Box::new(b2f) },
            }
        }
    }
}

// Constant folding for commands
pub open spec fn fold_constants_com(c: Com) -> Com
    decreases c
{
    match c {
        Com::CSkip => Com::CSkip,
        Com::CAsgn { x, a } => Com::CAsgn { x, a: Box::new(fold_constants_aexp(*a)) },
        Com::CSeq { c1, c2 } => Com::CSeq {
            c1: Box::new(fold_constants_com(*c1)),
            c2: Box::new(fold_constants_com(*c2)),
        },
        Com::CIf { b, ct, cf } => {
            let b_folded = fold_constants_bexp(*b);
            match b_folded {
                BExp::BTrue => fold_constants_com(*ct),
                BExp::BFalse => fold_constants_com(*cf),
                _ => Com::CIf {
                    b: Box::new(b_folded),
                    ct: Box::new(fold_constants_com(*ct)),
                    cf: Box::new(fold_constants_com(*cf)),
                },
            }
        }
        Com::CWhile { b, body } => {
            let b_folded = fold_constants_bexp(*b);
            match b_folded {
                BExp::BFalse => Com::CSkip,
                _ => Com::CWhile {
                    b: Box::new(b_folded),
                    body: Box::new(fold_constants_com(*body)),
                },
            }
        }
    }
}

// ============================================================================
// SOUNDNESS OF CONSTANT FOLDING
// ============================================================================

// Soundness definition for arithmetic expression transformations
pub open spec fn atrans_sound(atrans: spec_fn(AExp) -> AExp) -> bool {
    forall|a: AExp| #[trigger] aequiv(a, atrans(a))
}

// Soundness definition for boolean expression transformations
pub open spec fn btrans_sound(btrans: spec_fn(BExp) -> BExp) -> bool {
    forall|b: BExp| #[trigger] bequiv(b, btrans(b))
}

// Soundness definition for command transformations
pub open spec fn ctrans_sound(ctrans: spec_fn(Com) -> Com) -> bool {
    forall|c: Com| #[trigger] cequiv(c, ctrans(c))
}

// ============================================================================
// CONSTANT FOLDING SOUNDNESS
// ============================================================================

// Prove fold_constants_aexp preserves evaluation for a specific state
pub proof fn fold_constants_aexp_sound_st(a: AExp, st: Store)
    ensures aeval(a, st) == aeval(fold_constants_aexp(a), st)
    decreases a
{
    match a {
        AExp::ANum { n } => {}
        AExp::AId { x } => {}
        AExp::APlus { a1, a2 } => {
            fold_constants_aexp_sound_st(*a1, st);
            fold_constants_aexp_sound_st(*a2, st);
        }
        AExp::AMinus { a1, a2 } => {
            fold_constants_aexp_sound_st(*a1, st);
            fold_constants_aexp_sound_st(*a2, st);
        }
        AExp::AMult { a1, a2 } => {
            fold_constants_aexp_sound_st(*a1, st);
            fold_constants_aexp_sound_st(*a2, st);
        }
    }
}

// Prove fold_constants_bexp preserves evaluation for a specific state
pub proof fn fold_constants_bexp_sound_st(b: BExp, st: Store)
    ensures beval(b, st) == beval(fold_constants_bexp(b), st)
    decreases b
{
    match b {
        BExp::BTrue => {}
        BExp::BFalse => {}
        BExp::BEq { a1, a2 } => {
            fold_constants_aexp_sound_st(*a1, st);
            fold_constants_aexp_sound_st(*a2, st);
        }
        BExp::BLe { a1, a2 } => {
            fold_constants_aexp_sound_st(*a1, st);
            fold_constants_aexp_sound_st(*a2, st);
        }
        BExp::BNot { b1 } => {
            fold_constants_bexp_sound_st(*b1, st);
        }
        BExp::BAnd { b1, b2 } => {
            fold_constants_bexp_sound_st(*b1, st);
            fold_constants_bexp_sound_st(*b2, st);
        }
    }
}

// ============================================================================
// CONGRUENCE PROPERTIES
// ============================================================================

// Assignment congruence: if a1 ~ a1' then (x := a1) ~ (x := a1') for a specific fuel and state
pub proof fn CAsgn_congruence_fuel(x: Var, a1: AExp, a1_prime: AExp, fuel: nat, st: Store)
    requires
        fuel >= 1,
        aeval(a1, st) == aeval(a1_prime, st),
    ensures ceval(fuel, Com::CAsgn { x, a: Box::new(a1) }, st)
        == ceval(fuel, Com::CAsgn { x, a: Box::new(a1_prime) }, st)
{
    reveal_with_fuel(ceval, 2);
}

// ============================================================================
// EXAMPLE: SPECIFIC PROGRAM EQUIVALENCES
// ============================================================================

// Example: evaluating a simple addition
pub proof fn aexp_simple_add(st: Store)
    ensures aeval(AExp::APlus { a1: Box::new(AExp::ANum { n: 1 }), a2: Box::new(AExp::ANum { n: 2 }) }, st) == 3
{
    // Verus can compute this directly since aeval is open
    reveal_with_fuel(aeval, 3);
}

// Example: n - n = 0 (concrete proof)
pub proof fn aexp_n_minus_n(st: Store)
    ensures aeval(AExp::AMinus { a1: Box::new(AExp::ANum { n: 5 }), a2: Box::new(AExp::ANum { n: 5 }) }, st) == 0
{
    // Direct from aeval definition: 5 - 5 = 0
}

// Example: Constant folding preserves evaluation
pub proof fn fold_preserves_eval_example()
{
    let a = AExp::APlus {
        a1: Box::new(AExp::ANum { n: 1 }),
        a2: Box::new(AExp::ANum { n: 2 })
    };
    let st = store_empty();
    fold_constants_aexp_sound_st(a, st);
    // The folded version (ANum{3}) evaluates to the same value
}

// ============================================================================
// INEQUIVALENCE EXAMPLE
// ============================================================================

// Demonstration that not all programs are equivalent
// while true do skip diverges, while false do skip terminates
pub proof fn while_true_not_equiv_skip()
    ensures !cequiv(
        Com::CWhile { b: Box::new(BExp::BTrue), body: Box::new(Com::CSkip) },
        Com::CSkip
    )
{
    let st = store_empty();
    reveal_with_fuel(ceval, 2);
    // With fuel 1, skip returns Some(st), while (while true) returns None
    assert(ceval(1, Com::CSkip, st) == Option::Some(st));
    assert(ceval(1, Com::CWhile { b: Box::new(BExp::BTrue), body: Box::new(Com::CSkip) }, st) == Option::<Store>::None);
}

// ============================================================================
// MAIN VERIFICATION FUNCTION
// ============================================================================

pub proof fn equiv_examples_verify()
{
    // Test equivalence properties
    let a = AExp::ANum { n: 42 };
    aequiv_refl(a);

    let b = BExp::BTrue;
    bequiv_refl(b);

    let c = Com::CSkip;
    cequiv_refl(c);

    // Test simple equivalences with specific fuel
    let st = store_empty();
    skip_left_fuel2(Com::CSkip, 2, st);
    if_true_fuel(Com::CSkip, Com::CSkip, 2, st);
    if_false_fuel(Com::CSkip, Com::CSkip, 2, st);
    while_false_fuel(Com::CSkip, 1, st);
    skip_terminates(1, st);

    // Test constant folding soundness
    fold_constants_aexp_sound_st(AExp::APlus {
        a1: Box::new(AExp::ANum { n: 1 }),
        a2: Box::new(AExp::ANum { n: 2 })
    }, st);

    // Test specific examples
    aexp_simple_add(st);
    aexp_n_minus_n(st);
    fold_preserves_eval_example();

    // Test inequivalence
    while_true_not_equiv_skip();
}

pub fn main() {
    proof {
        equiv_examples_verify();
    }
}

} // verus!
