use vstd::prelude::*;
use vstd::seq_lib::*;
use vstd::seq::group_seq_axioms;

verus! {

broadcast use group_seq_axioms;

// Imp (Software Foundations, LF/Imp) in Verus.
// We model variables as `nat` indices (instead of strings).

pub type Var = nat;
pub type State = spec_fn(Var) -> int;

#[verifier::inline]
pub open spec fn empty_state() -> State {
    |v: Var| 0
}

#[verifier::inline]
pub open spec fn lookup(st: State, x: Var) -> int {
    st(x)
}

#[verifier::inline]
pub open spec fn update(st: State, x: Var, v: int) -> State {
    |y: Var| if y == x { v } else { st(y) }
}

pub enum AExp {
    N(int),
    V(Var),
    Plus(Seq<AExp>),
    Minus(Seq<AExp>),
    Mult(Seq<AExp>),
}

pub enum BExp {
    B(bool),
    Eq(Seq<AExp>),
    Le(Seq<AExp>),
    Not(Seq<BExp>),
    And(Seq<BExp>),
}

pub enum Com {
    Skip,
    Assign(Var, AExp),
    Seq(Seq<Com>),
    If(BExp, Seq<Com>),
    While(BExp, Seq<Com>),
}

#[verifier::inline]
pub open spec fn fuel_measure(fuel: int) -> nat {
    if fuel <= 0 {
        spec_literal_nat("0")
    } else {
        spec_cast_integer::<int, nat>(fuel)
    }
}

pub uninterp spec fn aeval(st: State, a: AExp, fuel: int) -> Option<int>;

pub uninterp spec fn beval(st: State, b: BExp, fuel: int) -> Option<bool>;

pub broadcast axiom fn axiom_aeval_N(st: State, n: int, fuel: int)
    requires fuel > 0,
    ensures #[trigger] aeval(st, AExp::N(n), fuel) == Some(n),
;

pub broadcast axiom fn axiom_aeval_V(st: State, x: Var, fuel: int)
    requires fuel > 0,
    ensures #[trigger] aeval(st, AExp::V(x), fuel) == Some(lookup(st, x)),
;

pub broadcast axiom fn axiom_aeval_Plus(st: State, args: Seq<AExp>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] aeval(st, AExp::Plus(args), fuel)
            == if args.len() == 2 {
                match aeval(st, args[0], fuel - 1) {
                    None => None,
                    Some(v1) => match aeval(st, args[1], fuel - 1) {
                        None => None,
                        Some(v2) => Some(v1 + v2),
                    },
                }
            } else {
                None
            },
;

pub broadcast axiom fn axiom_aeval_Minus(st: State, args: Seq<AExp>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] aeval(st, AExp::Minus(args), fuel)
            == if args.len() == 2 {
                match aeval(st, args[0], fuel - 1) {
                    None => None,
                    Some(v1) => match aeval(st, args[1], fuel - 1) {
                        None => None,
                        Some(v2) => Some(v1 - v2),
                    },
                }
            } else {
                None
            },
;

pub broadcast axiom fn axiom_aeval_Mult(st: State, args: Seq<AExp>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] aeval(st, AExp::Mult(args), fuel)
            == if args.len() == 2 {
                match aeval(st, args[0], fuel - 1) {
                    None => None,
                    Some(v1) => match aeval(st, args[1], fuel - 1) {
                        None => None,
                        Some(v2) => Some(v1 * v2),
                    },
                }
            } else {
                None
            },
;

pub broadcast axiom fn axiom_beval_B(st: State, v: bool, fuel: int)
    requires fuel > 0,
    ensures #[trigger] beval(st, BExp::B(v), fuel) == Some(v),
;

pub broadcast axiom fn axiom_beval_Eq(st: State, args: Seq<AExp>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] beval(st, BExp::Eq(args), fuel)
            == if args.len() == 2 {
                match aeval(st, args[0], fuel - 1) {
                    None => None,
                    Some(v1) => match aeval(st, args[1], fuel - 1) {
                        None => None,
                        Some(v2) => Some(v1 == v2),
                    },
                }
            } else {
                None
            },
;

pub broadcast axiom fn axiom_beval_Le(st: State, args: Seq<AExp>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] beval(st, BExp::Le(args), fuel)
            == if args.len() == 2 {
                match aeval(st, args[0], fuel - 1) {
                    None => None,
                    Some(v1) => match aeval(st, args[1], fuel - 1) {
                        None => None,
                        Some(v2) => Some(v1 <= v2),
                    },
                }
            } else {
                None
            },
;

pub broadcast axiom fn axiom_beval_Not(st: State, args: Seq<BExp>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] beval(st, BExp::Not(args), fuel)
            == if args.len() == 1 {
                match beval(st, args[0], fuel - 1) {
                    None => None,
                    Some(v) => Some(!v),
                }
            } else {
                None
            },
;

pub broadcast axiom fn axiom_beval_And(st: State, args: Seq<BExp>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] beval(st, BExp::And(args), fuel)
            == if args.len() == 2 {
                match beval(st, args[0], fuel - 1) {
                    None => None,
                    Some(v1) => match beval(st, args[1], fuel - 1) {
                        None => None,
                        Some(v2) => Some(v1 && v2),
                    },
                }
            } else {
                None
            },
;


// Big-step evaluator with fuel.
// This matches the standard while-as-seq unfolding used in LF.
pub uninterp spec fn ceval(c: Com, st: State, fuel: int) -> Option<State>;

pub broadcast axiom fn axiom_ceval_Skip(st: State, fuel: int)
    requires fuel > 0,
    ensures #[trigger] ceval(Com::Skip, st, fuel) == Some(st),
;

pub broadcast axiom fn axiom_ceval_Assign(st: State, x: Var, a: AExp, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] ceval(Com::Assign(x, a), st, fuel)
            == match aeval(st, a, fuel - 1) {
                None => None,
                Some(v) => Some(update(st, x, v)),
            },
;

pub broadcast axiom fn axiom_ceval_Seq(st: State, cs: Seq<Com>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] ceval(Com::Seq(cs), st, fuel)
            == if cs.len() == 2 {
                match ceval(cs[0], st, fuel - 1) {
                    None => None,
                    Some(st1) => ceval(cs[1], st1, fuel - 1),
                }
            } else {
                None
            },
;

pub broadcast axiom fn axiom_ceval_If(st: State, b: BExp, branches: Seq<Com>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] ceval(Com::If(b, branches), st, fuel)
            == if branches.len() == 2 {
                match beval(st, b, fuel - 1) {
                    None => None,
                    Some(cond) => {
                        if cond {
                            ceval(branches[0], st, fuel - 1)
                        } else {
                            ceval(branches[1], st, fuel - 1)
                        }
                    }
                }
            } else {
                None
            },
;

pub broadcast axiom fn axiom_ceval_While(st: State, b: BExp, body: Seq<Com>, fuel: int)
    requires fuel > 0,
    ensures
        #[trigger] ceval(Com::While(b, body), st, fuel)
            == if body.len() == 1 {
                match beval(st, b, fuel - 1) {
                    None => None,
                    Some(cond) => {
                        if cond {
                            let c_next = Com::Seq(seq![body[0], Com::While(b, body)]);
                            ceval(c_next, st, fuel - 1)
                        } else {
                            Some(st)
                        }
                    }
                }
            } else {
                None
            },
;

pub open spec fn opt_is_some<T>(o: Option<T>) -> bool {
    match o {
        None => false,
        Some(_) => true,
    }
}

// Some variable names (as indices) for examples.
pub const X: Var = spec_literal_nat("0");
pub const Y: Var = spec_literal_nat("1");
pub const Z: Var = spec_literal_nat("2");

fn main() {
    proof {
        imp_examples_verify();
    }
}

pub proof fn imp_examples_verify() {
    broadcast use {
        axiom_aeval_N,
        axiom_aeval_V,
        axiom_aeval_Plus,
        axiom_aeval_Minus,
        axiom_aeval_Mult,
        axiom_beval_B,
        axiom_beval_Eq,
        axiom_beval_Le,
        axiom_beval_Not,
        axiom_beval_And,
        axiom_ceval_Skip,
        axiom_ceval_Assign,
        axiom_ceval_Seq,
        axiom_ceval_If,
        axiom_ceval_While,
    };

    // Example 1: Z := X + Y
    let st0 = update(update(empty_state(), X, 2int), Y, 3int);
    let args_xy = seq![AExp::V(X), AExp::V(Y)];
    assert(args_xy.len() == 2);
    assert(args_xy[0] == AExp::V(X));
    assert(args_xy[1] == AExp::V(Y));
    let prog1 = Com::Assign(Z, AExp::Plus(seq![AExp::V(X), AExp::V(Y)]));
    let r1 = ceval(prog1, st0, 10);
    assert(r1 is Some);
    assert(lookup(st0, X) == 2int);
    assert(lookup(st0, Y) == 3int);
    assert(aeval(st0, AExp::Plus(seq![AExp::V(X), AExp::V(Y)]), 10) == Some(5int));
    assert(r1 == Some(update(st0, Z, 5int)));
    let st1 = r1->Some_0;
    assert(lookup(st1, Z) == 5int);

    // Example 2: if X <= Y then Z := 1 else Z := 0
    let st2 = update(update(empty_state(), X, 10int), Y, 20int);
    let prog2 = Com::If(
        BExp::Le(seq![AExp::V(X), AExp::V(Y)]),
        seq![Com::Assign(Z, AExp::N(1int)), Com::Assign(Z, AExp::N(0int))],
    );
    let r2 = ceval(prog2, st2, 10);
    assert(beval(st2, BExp::Le(seq![AExp::V(X), AExp::V(Y)]), 10) == Some(true));
    assert(r2 == Some(update(st2, Z, 1int)));
    assert(r2 is Some);
    let st3 = r2->Some_0;
    assert(lookup(st3, Z) == 1int);

    // Example 3: while X <= 2 do X := X + 1 end
    // Starting from X=0, this should terminate with X=3.
    let st3 = update(empty_state(), X, 0int);
    let body = Com::Assign(X, AExp::Plus(seq![AExp::V(X), AExp::N(1int)]));
    let prog3 = Com::While(BExp::Le(seq![AExp::V(X), AExp::N(2int)]), seq![body]);
    let r3 = ceval(prog3, st3, 50);
    assert(r3 is Some);
    let st4 = r3->Some_0;
    assert(lookup(st4, X) == 3int);
}

} // verus!
