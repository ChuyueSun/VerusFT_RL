use vstd::prelude::*;
use vstd::map::*;
use vstd::set::*;

verus! {

// ImpCEvalFun (Software Foundations, LF/ImpCEvalFun) style snippets in Verus.
//
// We define a *functional* evaluator for a tiny IMP language using a fuel parameter,
// returning `Option<State>` to model potential nontermination when fuel runs out.

broadcast use { group_map_axioms, group_set_axioms };

pub type Var = nat;

pub type Store = Map<Var, int>;

pub open spec fn store_empty() -> Store {
    Map::<Var, int>::empty()
}

pub open spec fn store_update(st: Store, x: Var, v: int) -> Store {
    st.insert(x, v)
}

pub open spec fn store_get(st: Store, default: int, x: Var) -> int {
    if st.dom().contains(x) { st[x] } else { default }
}

// ----------------------------
// Expressions
// ----------------------------

pub enum AExp {
    N { n: int },
    V { x: Var },
    Plus { a1: Box<AExp>, a2: Box<AExp> },
    Minus { a1: Box<AExp>, a2: Box<AExp> },
}

pub enum BExp {
    B { b: bool },
    Eq { a1: Box<AExp>, a2: Box<AExp> },
    Le { a1: Box<AExp>, a2: Box<AExp> },
    Not { b1: Box<BExp> },
    And { b1: Box<BExp>, b2: Box<BExp> },
}

pub open spec fn aeval(a: AExp, st: Store, default: int) -> int
    decreases a
{
    match a {
        AExp::N { n } => n,
        AExp::V { x } => store_get(st, default, x),
        AExp::Plus { a1, a2 } => aeval(*a1, st, default) + aeval(*a2, st, default),
        AExp::Minus { a1, a2 } => aeval(*a1, st, default) - aeval(*a2, st, default),
    }
}

pub open spec fn beval(b: BExp, st: Store, default: int) -> bool
    decreases b
{
    match b {
        BExp::B { b } => b,
        BExp::Eq { a1, a2 } => aeval(*a1, st, default) == aeval(*a2, st, default),
        BExp::Le { a1, a2 } => aeval(*a1, st, default) <= aeval(*a2, st, default),
        BExp::Not { b1 } => !beval(*b1, st, default),
        BExp::And { b1, b2 } => beval(*b1, st, default) && beval(*b2, st, default),
    }
}

// ----------------------------
// Commands
// ----------------------------

pub enum Com {
    Skip,
    Assign { x: Var, a: Box<AExp> },
    Seq { c1: Box<Com>, c2: Box<Com> },
    If { b: Box<BExp>, ct: Box<Com>, cf: Box<Com> },
    While { b: Box<BExp>, body: Box<Com> },
}

pub open spec fn ceval_step(fuel: nat, c: Com, st: Store, default: int) -> Option<Store>
    decreases fuel, c
{
    if fuel == 0 {
        Option::<Store>::None
    } else {
        let fuel1 = (fuel - 1) as nat;
        match c {
            Com::Skip => Option::Some(st),
            Com::Assign { x, a } => Option::Some(store_update(st, x, aeval(*a, st, default))),
            Com::Seq { c1, c2 } => {
                match ceval_step(fuel1, *c1, st, default) {
                    Option::None => Option::<Store>::None,
                    Option::Some(st1) => ceval_step(fuel1, *c2, st1, default),
                }
            }
            Com::If { b, ct, cf } => {
                if beval(*b, st, default) {
                    ceval_step(fuel1, *ct, st, default)
                } else {
                    ceval_step(fuel1, *cf, st, default)
                }
            }
            Com::While { b, body } => {
                if beval(*b, st, default) {
                    match ceval_step(fuel1, *body, st, default) {
                        Option::None => Option::<Store>::None,
                        Option::Some(st1) => ceval_step(fuel1, Com::While { b: Box::new(*b), body: Box::new(*body) }, st1, default),
                    }
                } else {
                    Option::Some(st)
                }
            }
        }
    }
}

// ----------------------------
// Lemmas / examples
// ----------------------------

pub proof fn lemma_ceval_step_skip(fuel: nat, st: Store, default: int)
    requires fuel > 0
    ensures ceval_step(fuel, Com::Skip, st, default) == Option::Some(st)
{
    reveal_with_fuel(ceval_step, 1);
}

pub proof fn lemma_ceval_step_assign(fuel: nat, st: Store, default: int, x: Var, a: AExp)
    requires fuel > 0
    ensures ceval_step(fuel, Com::Assign { x, a: Box::new(a) }, st, default)
        == Option::Some(store_update(st, x, aeval(a, st, default)))
{
    reveal_with_fuel(ceval_step, 1);
}

pub proof fn lemma_ceval_step_seq_skip_l(fuel: nat, c2: Com, st: Store, default: int)
    requires fuel > 0
    ensures ceval_step(fuel, Com::Seq { c1: Box::new(Com::Skip), c2: Box::new(c2) }, st, default)
        == ceval_step((fuel - 1) as nat, c2, st, default)
{
    let fuel1 = (fuel - 1) as nat;
    reveal_with_fuel(ceval_step, 1);
    if fuel1 == 0 {
        // body of seq matches on ceval_step(0, Skip, ...) which is None
        reveal_with_fuel(ceval_step, 1);
    } else {
        lemma_ceval_step_skip(fuel1, st, default);
        // now the seq reduces to the second command with fuel1
        reveal_with_fuel(ceval_step, 1);
    }
}

// Divergence demo (fuel-bounded): even with fuel=1, `while true do skip` cannot finish.
pub proof fn lemma_while_true_skip_none_fuel0(st: Store, default: int)
    ensures ceval_step(0, Com::While { b: Box::new(BExp::B { b: true }), body: Box::new(Com::Skip) }, st, default)
        == Option::<Store>::None
{
    reveal_with_fuel(ceval_step, 1);
}

pub proof fn lemma_while_true_skip_none_fuel1(st: Store, default: int)
    ensures ceval_step(1, Com::While { b: Box::new(BExp::B { b: true }), body: Box::new(Com::Skip) }, st, default)
        == Option::<Store>::None
{
    reveal_with_fuel(ceval_step, 2);
}

pub proof fn ex_basic_sanity()
{
    let default: int = 0;
    let st0 = store_empty();

    lemma_ceval_step_skip(1, st0, default);
    lemma_ceval_step_assign(1, st0, default, 0, AExp::N { n: 41 });

    let st1 = store_update(st0, 0, 41);
    assert(store_get(st1, default, 0) == 41);
}

pub fn main() {
    proof {
        ex_basic_sanity();
        lemma_while_true_skip_none_fuel0(store_empty(), 0);
        lemma_while_true_skip_none_fuel1(store_empty(), 0);
    }
}

} // verus!
