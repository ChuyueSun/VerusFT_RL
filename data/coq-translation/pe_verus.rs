use vstd::prelude::*;
use vstd::seq::*;

verus! {

// ============================================================================
// Partial Evaluation (Software Foundations, PLF/PE)
// Specialization, static vs dynamic, residual programs
// ============================================================================

// ----------------------------------------------------------------------------
// Variables and State
// ----------------------------------------------------------------------------

pub type Var = nat;

pub spec const X: Var = 0;
pub spec const Y: Var = 1;
pub spec const Z: Var = 2;

// Partial state: list of (var, value) pairs for known variables
pub type PEState = Seq<(Var, int)>;

pub open spec fn empty_pe_state() -> PEState {
    Seq::<(Var, int)>::empty()
}

// Lookup a variable in partial state
pub open spec fn pe_lookup(st: PEState, x: Var) -> Option<int>
    decreases st.len()
{
    if st.len() == 0 {
        Option::None
    } else {
        let (v, val) = st[0];
        if v == x {
            Option::Some(val)
        } else {
            pe_lookup(st.skip(1), x)
        }
    }
}

// Extend partial state
pub open spec fn pe_add(st: PEState, x: Var, v: int) -> PEState {
    Seq::<(Var, int)>::empty().push((x, v)).add(st)
}

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

// Partially evaluate an arithmetic expression
// Returns either a constant (if fully known) or a residual expression
pub enum PEAExp {
    Static { n: int },           // Fully evaluated constant
    Dynamic { a: AExp },         // Residual expression
}

pub open spec fn pe_aexp(st: PEState, a: AExp) -> PEAExp
    decreases a
{
    match a {
        AExp::ANum { n } => PEAExp::Static { n },
        AExp::AId { x } => {
            match pe_lookup(st, x) {
                Option::Some(v) => PEAExp::Static { n: v },
                Option::None => PEAExp::Dynamic { a },
            }
        }
        AExp::APlus { a1, a2 } => {
            match (pe_aexp(st, *a1), pe_aexp(st, *a2)) {
                (PEAExp::Static { n: n1 }, PEAExp::Static { n: n2 }) =>
                    PEAExp::Static { n: n1 + n2 },
                (PEAExp::Static { n: n1 }, PEAExp::Dynamic { a: a2 }) =>
                    PEAExp::Dynamic { a: AExp::APlus { a1: Box::new(AExp::ANum { n: n1 }), a2: Box::new(a2) } },
                (PEAExp::Dynamic { a: a1 }, PEAExp::Static { n: n2 }) =>
                    PEAExp::Dynamic { a: AExp::APlus { a1: Box::new(a1), a2: Box::new(AExp::ANum { n: n2 }) } },
                (PEAExp::Dynamic { a: a1 }, PEAExp::Dynamic { a: a2 }) =>
                    PEAExp::Dynamic { a: AExp::APlus { a1: Box::new(a1), a2: Box::new(a2) } },
            }
        }
        AExp::AMinus { a1, a2 } => {
            match (pe_aexp(st, *a1), pe_aexp(st, *a2)) {
                (PEAExp::Static { n: n1 }, PEAExp::Static { n: n2 }) =>
                    PEAExp::Static { n: n1 - n2 },
                _ => PEAExp::Dynamic { a },  // Simplified
            }
        }
        AExp::AMult { a1, a2 } => {
            match (pe_aexp(st, *a1), pe_aexp(st, *a2)) {
                (PEAExp::Static { n: n1 }, PEAExp::Static { n: n2 }) =>
                    PEAExp::Static { n: n1 * n2 },
                _ => PEAExp::Dynamic { a },  // Simplified
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Boolean Expressions
// ----------------------------------------------------------------------------

pub enum BExp {
    BTrue,
    BFalse,
    BEq { a1: AExp, a2: AExp },
    BLe { a1: AExp, a2: AExp },
    BNot { b: Box<BExp> },
    BAnd { b1: Box<BExp>, b2: Box<BExp> },
}

pub enum PEBExp {
    Static { b: bool },
    Dynamic { b: BExp },
}

pub open spec fn pe_bexp(st: PEState, b: BExp) -> PEBExp
    decreases b
{
    match b {
        BExp::BTrue => PEBExp::Static { b: true },
        BExp::BFalse => PEBExp::Static { b: false },
        BExp::BEq { a1, a2 } => {
            match (pe_aexp(st, a1), pe_aexp(st, a2)) {
                (PEAExp::Static { n: n1 }, PEAExp::Static { n: n2 }) =>
                    PEBExp::Static { b: n1 == n2 },
                _ => PEBExp::Dynamic { b },
            }
        }
        BExp::BLe { a1, a2 } => {
            match (pe_aexp(st, a1), pe_aexp(st, a2)) {
                (PEAExp::Static { n: n1 }, PEAExp::Static { n: n2 }) =>
                    PEBExp::Static { b: n1 <= n2 },
                _ => PEBExp::Dynamic { b },
            }
        }
        BExp::BNot { b: inner } => {
            match pe_bexp(st, *inner) {
                PEBExp::Static { b: v } => PEBExp::Static { b: !v },
                PEBExp::Dynamic { b: b_dyn } =>
                    PEBExp::Dynamic { b: BExp::BNot { b: Box::new(b_dyn) } },
            }
        }
        BExp::BAnd { b1, b2 } => {
            match pe_bexp(st, *b1) {
                PEBExp::Static { b: false } => PEBExp::Static { b: false },
                PEBExp::Static { b: true } => pe_bexp(st, *b2),
                PEBExp::Dynamic { b: b1_dyn } => PEBExp::Dynamic { b },  // Simplified
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Commands
// ----------------------------------------------------------------------------

pub enum Com {
    CSkip,
    CAsgn { x: Var, a: AExp },
    CSeq { c1: Box<Com>, c2: Box<Com> },
    CIf { b: BExp, c1: Box<Com>, c2: Box<Com> },
}

// Residual command after partial evaluation
pub enum ResCom {
    RSkip,
    RAsgn { x: Var, a: AExp },
    RSeq { c1: Box<ResCom>, c2: Box<ResCom> },
    RIf { b: BExp, c1: Box<ResCom>, c2: Box<ResCom> },
}

// Result of partial evaluation: new state + residual command
pub struct PEResult {
    pub st: PEState,
    pub residual: ResCom,
}

// ----------------------------------------------------------------------------
// Examples: Arithmetic Partial Evaluation
// ----------------------------------------------------------------------------

// PE of constant: 5 => Static(5)
pub proof fn example_pe_const()
{
    let st = empty_pe_state();
    let a = AExp::ANum { n: 5 };
    assert(pe_aexp(st, a) == (PEAExp::Static { n: 5 }));
}

// PE of known variable: X where X=3 => Static(3)
pub proof fn example_pe_known_var()
{
    let st = pe_add(empty_pe_state(), X, 3);
    let a = AExp::AId { x: X };

    // Need to show pe_lookup finds X=3
    assert(st[0] == (X, 3int));
    reveal_with_fuel(pe_lookup, 2);
    assert(pe_lookup(st, X) == Option::Some(3int));
    assert(pe_aexp(st, a) == (PEAExp::Static { n: 3 }));
}

// PE of unknown variable: Y where only X is known => Dynamic(Y)
pub proof fn example_pe_unknown_var()
{
    let st = pe_add(empty_pe_state(), X, 3);
    let a = AExp::AId { x: Y };

    reveal_with_fuel(pe_lookup, 2);
    assert(pe_lookup(st, Y) == Option::<int>::None);
    assert(pe_aexp(st, a) == (PEAExp::Dynamic { a }));
}

// PE of 3 + 5 => Static(8)
pub proof fn example_pe_add_static()
{
    let st = empty_pe_state();
    let a = AExp::APlus {
        a1: Box::new(AExp::ANum { n: 3 }),
        a2: Box::new(AExp::ANum { n: 5 })
    };
    reveal_with_fuel(pe_aexp, 3);
    assert(pe_aexp(st, a) == (PEAExp::Static { n: 8 }));
}

// PE of X + 1 where X=3 => Static(4)
pub proof fn example_pe_add_known()
{
    let st = pe_add(empty_pe_state(), X, 3);
    let a = AExp::APlus {
        a1: Box::new(AExp::AId { x: X }),
        a2: Box::new(AExp::ANum { n: 1 })
    };

    reveal_with_fuel(pe_lookup, 2);
    reveal_with_fuel(pe_aexp, 3);
    assert(pe_aexp(st, a) == (PEAExp::Static { n: 4 }));
}

// ----------------------------------------------------------------------------
// Examples: Boolean Partial Evaluation
// ----------------------------------------------------------------------------

// PE of true => Static(true)
pub proof fn example_pe_btrue()
{
    let st = empty_pe_state();
    assert(pe_bexp(st, BExp::BTrue) == (PEBExp::Static { b: true }));
}

// PE of 3 == 3 => Static(true)
pub proof fn example_pe_beq_true()
{
    let st = empty_pe_state();
    let b = BExp::BEq {
        a1: AExp::ANum { n: 3 },
        a2: AExp::ANum { n: 3 }
    };
    assert(pe_bexp(st, b) == (PEBExp::Static { b: true }));
}

// PE of 3 == 5 => Static(false)
pub proof fn example_pe_beq_false()
{
    let st = empty_pe_state();
    let b = BExp::BEq {
        a1: AExp::ANum { n: 3 },
        a2: AExp::ANum { n: 5 }
    };
    assert(pe_bexp(st, b) == (PEBExp::Static { b: false }));
}

// PE of !true => Static(false)
pub proof fn example_pe_bnot()
{
    let st = empty_pe_state();
    let b = BExp::BNot { b: Box::new(BExp::BTrue) };
    reveal_with_fuel(pe_bexp, 2);
    assert(pe_bexp(st, b) == (PEBExp::Static { b: false }));
}

// PE of false && anything => Static(false) (short-circuit)
pub proof fn example_pe_band_short()
{
    let st = empty_pe_state();
    let b = BExp::BAnd {
        b1: Box::new(BExp::BFalse),
        b2: Box::new(BExp::BTrue)
    };
    reveal_with_fuel(pe_bexp, 2);
    assert(pe_bexp(st, b) == (PEBExp::Static { b: false }));
}

// ----------------------------------------------------------------------------
// Examples: Specialization Concept
// ----------------------------------------------------------------------------

// Specializing "X := 3; Y := X + 1" with empty state
// After PE: X is known (3), Y becomes X + 1 = 4
pub proof fn example_specialization_concept()
{
    // After X := 3, we know X = 3
    let st_after_x = pe_add(empty_pe_state(), X, 3);

    // When evaluating Y := X + 1, X is known
    let a = AExp::APlus {
        a1: Box::new(AExp::AId { x: X }),
        a2: Box::new(AExp::ANum { n: 1 })
    };

    reveal_with_fuel(pe_lookup, 2);
    reveal_with_fuel(pe_aexp, 3);
    // X + 1 where X=3 evaluates to 4 statically
    assert(pe_aexp(st_after_x, a) == (PEAExp::Static { n: 4 }));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn pe_examples_verify()
{
    // Arithmetic PE
    example_pe_const();
    example_pe_known_var();
    example_pe_unknown_var();
    example_pe_add_static();
    example_pe_add_known();

    // Boolean PE
    example_pe_btrue();
    example_pe_beq_true();
    example_pe_beq_false();
    example_pe_bnot();
    example_pe_band_short();

    // Specialization
    example_specialization_concept();
}

pub fn main() {
    proof {
        pe_examples_verify();
    }
}

} // verus!
