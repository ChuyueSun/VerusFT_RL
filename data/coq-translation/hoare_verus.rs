use vstd::prelude::*;
use vstd::map::*;
use vstd::set::*;

verus! {

// ============================================================================
// Hoare Logic, Part I (Software Foundations, PLF/Hoare)
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

pub spec const X: Var = 0;
pub spec const Y: Var = 1;
pub spec const Z: Var = 2;

// ----------------------------------------------------------------------------
// Arithmetic and Boolean Expressions
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
// ASSERTIONS
// ============================================================================

pub type Assertion = spec_fn(Store) -> bool;

pub open spec fn assert_true() -> Assertion {
    |st: Store| true
}

pub open spec fn assert_false() -> Assertion {
    |st: Store| false
}

pub open spec fn assert_eq_var(x: Var, v: int) -> Assertion {
    |st: Store| store_get(st, x) == v
}

// ============================================================================
// HOARE TRIPLE DEFINITION
// ============================================================================

// Partial correctness: if precondition holds and command terminates,
// then postcondition holds
pub open spec fn valid_hoare_triple(p: Assertion, c: Com, q: Assertion) -> bool {
    forall|fuel: nat, st: Store, st_final: Store|
        #![trigger ceval(fuel, c, st), p(st), q(st_final)]
        (p(st) && ceval(fuel, c, st) == Option::Some(st_final)) ==> q(st_final)
}

// ============================================================================
// HOARE LOGIC RULES - VERIFIED LEMMAS
// ============================================================================

// Skip rule: {P} skip {P}
pub proof fn hoare_skip(fuel: nat, st: Store)
    requires fuel > 0
    ensures ceval(fuel, Com::CSkip, st) == Option::Some(st)
{
    reveal_with_fuel(ceval, 1);
}

// Assignment semantics
pub proof fn hoare_asgn_semantics(fuel: nat, st: Store, x: Var, a: AExp)
    requires fuel > 0
    ensures ceval(fuel, Com::CAsgn { x, a: Box::new(a) }, st)
        == Option::Some(store_update(st, x, aeval(a, st)))
{
    reveal_with_fuel(ceval, 1);
}

// Sequence semantics
pub proof fn hoare_seq_semantics(fuel: nat, st: Store, c1: Com, c2: Com, st1: Store)
    requires
        fuel > 1,
        ceval((fuel - 1) as nat, c1, st) == Option::Some(st1),
    ensures ceval(fuel, Com::CSeq { c1: Box::new(c1), c2: Box::new(c2) }, st)
        == ceval((fuel - 1) as nat, c2, st1)
{
    reveal_with_fuel(ceval, 2);
}

// If-true semantics
pub proof fn hoare_if_true_semantics(fuel: nat, st: Store, b: BExp, ct: Com, cf: Com)
    requires
        fuel > 0,
        beval(b, st),
    ensures ceval(fuel, Com::CIf { b: Box::new(b), ct: Box::new(ct), cf: Box::new(cf) }, st)
        == ceval((fuel - 1) as nat, ct, st)
{
    reveal_with_fuel(ceval, 2);
}

// If-false semantics
pub proof fn hoare_if_false_semantics(fuel: nat, st: Store, b: BExp, ct: Com, cf: Com)
    requires
        fuel > 0,
        !beval(b, st),
    ensures ceval(fuel, Com::CIf { b: Box::new(b), ct: Box::new(ct), cf: Box::new(cf) }, st)
        == ceval((fuel - 1) as nat, cf, st)
{
    reveal_with_fuel(ceval, 2);
}

// While-false terminates
pub proof fn hoare_while_false_semantics(fuel: nat, st: Store, b: BExp, body: Com)
    requires
        fuel > 0,
        !beval(b, st),
    ensures ceval(fuel, Com::CWhile { b: Box::new(b), body: Box::new(body) }, st)
        == Option::Some(st)
{
    reveal_with_fuel(ceval, 2);
}

// ============================================================================
// CONCRETE EXAMPLES
// ============================================================================

// Example: X := 5 sets X to 5
pub proof fn example_assign_5()
{
    let st = store_empty();
    reveal_with_fuel(ceval, 1);
    let result = ceval(1, Com::CAsgn { x: X, a: Box::new(AExp::ANum { n: 5 }) }, st);
    assert(result == Option::Some(store_update(st, X, 5)));
    assert(store_get(store_update(st, X, 5), X) == 5);
}

// Example: skip preserves state
pub proof fn example_skip_preserves()
{
    let st = store_update(store_empty(), X, 42);
    reveal_with_fuel(ceval, 1);
    assert(ceval(1, Com::CSkip, st) == Option::Some(st));
    assert(store_get(st, X) == 42);
}

// Example: X := X + 1 increments X
pub proof fn example_increment()
{
    let st = store_update(store_empty(), X, 2);
    reveal_with_fuel(ceval, 1);
    let cmd = Com::CAsgn { x: X, a: Box::new(AExp::APlus {
        a1: Box::new(AExp::AId { x: X }),
        a2: Box::new(AExp::ANum { n: 1 })
    })};
    let result = ceval(1, cmd, st);
    assert(aeval(AExp::AId { x: X }, st) == 2);
    assert(aeval(AExp::ANum { n: 1 }, st) == 1);
    assert(result == Option::Some(store_update(st, X, 3)));
}

// Example: Sequential composition - two assignments
pub proof fn example_sequence()
{
    let st = store_empty();
    reveal_with_fuel(ceval, 2);

    // First assignment: X := 5
    let c1 = Com::CAsgn { x: X, a: Box::new(AExp::ANum { n: 5 }) };
    let st1 = store_update(st, X, 5);
    assert(ceval(1, c1, st) == Option::Some(st1));
    assert(store_get(st1, X) == 5);

    // Second assignment: Y := 10
    let c2 = Com::CAsgn { x: Y, a: Box::new(AExp::ANum { n: 10 }) };
    let st2 = store_update(st1, Y, 10);
    assert(ceval(1, c2, st1) == Option::Some(st2));
    assert(store_get(st2, Y) == 10);
    assert(store_get(st2, X) == 5);  // X unchanged
}

// Example: If-then-else with true condition
pub proof fn example_if_true()
{
    let st = store_update(store_empty(), X, 10);
    reveal_with_fuel(ceval, 2);

    let cmd = Com::CIf {
        b: Box::new(BExp::BLe {
            a1: Box::new(AExp::AId { x: X }),
            a2: Box::new(AExp::ANum { n: 20 })
        }),
        ct: Box::new(Com::CAsgn { x: Y, a: Box::new(AExp::ANum { n: 1 }) }),
        cf: Box::new(Com::CAsgn { x: Y, a: Box::new(AExp::ANum { n: 0 }) }),
    };

    assert(beval(BExp::BLe {
        a1: Box::new(AExp::AId { x: X }),
        a2: Box::new(AExp::ANum { n: 20 })
    }, st) == true);

    let result = ceval(2, cmd, st);
    assert(result == Option::Some(store_update(st, Y, 1)));
}

// Example: While loop that terminates immediately
pub proof fn example_while_false()
{
    let st = store_update(store_empty(), X, 10);
    reveal_with_fuel(ceval, 2);

    // while X <= 5 do X := X + 1 (condition false, terminates)
    let cmd = Com::CWhile {
        b: Box::new(BExp::BLe {
            a1: Box::new(AExp::AId { x: X }),
            a2: Box::new(AExp::ANum { n: 5 })
        }),
        body: Box::new(Com::CAsgn { x: X, a: Box::new(AExp::APlus {
            a1: Box::new(AExp::AId { x: X }),
            a2: Box::new(AExp::ANum { n: 1 })
        })})
    };

    assert(beval(BExp::BLe {
        a1: Box::new(AExp::AId { x: X }),
        a2: Box::new(AExp::ANum { n: 5 })
    }, st) == false);

    assert(ceval(2, cmd, st) == Option::Some(st));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn hoare_examples_verify()
{
    example_assign_5();
    example_skip_preserves();
    example_increment();
    example_sequence();
    example_if_true();
    example_while_false();
}

pub fn main() {
    proof {
        hoare_examples_verify();
    }
}

} // verus!
