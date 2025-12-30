use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// Properties of STLC (Software Foundations, PLF/StlcProp)
// Progress, Preservation, and Type Soundness
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

pub enum Ty {
    TBool,
    TArrow { t1: Box<Ty>, t2: Box<Ty> },
}

// ----------------------------------------------------------------------------
// Terms
// ----------------------------------------------------------------------------

pub type Var = nat;

pub enum Tm {
    Var { x: Var },
    Abs { x: Var, ty: Ty, body: Box<Tm> },
    App { t1: Box<Tm>, t2: Box<Tm> },
    Tru,
    Fls,
    Ite { cond: Box<Tm>, then_br: Box<Tm>, else_br: Box<Tm> },
}

pub spec const X: Var = 0;
pub spec const Y: Var = 1;

// ----------------------------------------------------------------------------
// Values
// ----------------------------------------------------------------------------

pub open spec fn value(t: Tm) -> bool {
    match t {
        Tm::Abs { .. } => true,
        Tm::Tru => true,
        Tm::Fls => true,
        _ => false,
    }
}

// ----------------------------------------------------------------------------
// Substitution
// ----------------------------------------------------------------------------

pub open spec fn subst(x: Var, s: Tm, t: Tm) -> Tm
    decreases t
{
    match t {
        Tm::Var { x: y } => if x == y { s } else { t },
        Tm::Abs { x: y, ty, body } =>
            if x == y { t }
            else { Tm::Abs { x: y, ty, body: Box::new(subst(x, s, *body)) } },
        Tm::App { t1, t2 } =>
            Tm::App { t1: Box::new(subst(x, s, *t1)), t2: Box::new(subst(x, s, *t2)) },
        Tm::Tru => Tm::Tru,
        Tm::Fls => Tm::Fls,
        Tm::Ite { cond, then_br, else_br } =>
            Tm::Ite {
                cond: Box::new(subst(x, s, *cond)),
                then_br: Box::new(subst(x, s, *then_br)),
                else_br: Box::new(subst(x, s, *else_br)),
            },
    }
}

// ----------------------------------------------------------------------------
// Small-step Semantics
// ----------------------------------------------------------------------------

pub open spec fn step(t: Tm) -> Option<Tm>
    decreases t
{
    match t {
        Tm::App { t1, t2 } => {
            match *t1 {
                Tm::Abs { x, ty: _, body } if value(*t2) =>
                    Option::Some(subst(x, *t2, *body)),
                _ => {
                    if !value(*t1) {
                        match step(*t1) {
                            Option::Some(t1p) =>
                                Option::Some(Tm::App { t1: Box::new(t1p), t2: t2 }),
                            Option::None => Option::None,
                        }
                    } else if !value(*t2) {
                        match step(*t2) {
                            Option::Some(t2p) =>
                                Option::Some(Tm::App { t1: t1, t2: Box::new(t2p) }),
                            Option::None => Option::None,
                        }
                    } else {
                        Option::None
                    }
                }
            }
        }
        Tm::Ite { cond, then_br, else_br } => {
            match *cond {
                Tm::Tru => Option::Some(*then_br),
                Tm::Fls => Option::Some(*else_br),
                _ => match step(*cond) {
                    Option::Some(cp) =>
                        Option::Some(Tm::Ite { cond: Box::new(cp), then_br, else_br }),
                    Option::None => Option::None,
                }
            }
        }
        _ => Option::None,
    }
}

// ----------------------------------------------------------------------------
// Typing Context
// ----------------------------------------------------------------------------

pub type Context = Map<Var, Ty>;

pub open spec fn empty_ctx() -> Context {
    Map::<Var, Ty>::empty()
}

pub open spec fn ctx_extend(ctx: Context, x: Var, ty: Ty) -> Context {
    ctx.insert(x, ty)
}

// ----------------------------------------------------------------------------
// Typing Relation
// ----------------------------------------------------------------------------

pub open spec fn has_type(ctx: Context, t: Tm, ty: Ty) -> bool
    decreases t
{
    match t {
        Tm::Var { x } => ctx.dom().contains(x) && ctx[x] == ty,
        Tm::Abs { x, ty: ty_param, body } => {
            match ty {
                Ty::TArrow { t1, t2 } =>
                    *t1 == ty_param &&
                    has_type(ctx_extend(ctx, x, ty_param), *body, *t2),
                _ => false,
            }
        }
        Tm::App { t1, t2 } => {
            exists|ty_arg: Ty| #![auto]
                has_type(ctx, *t1, Ty::TArrow { t1: Box::new(ty_arg), t2: Box::new(ty) }) &&
                has_type(ctx, *t2, ty_arg)
        }
        Tm::Tru => ty == Ty::TBool,
        Tm::Fls => ty == Ty::TBool,
        Tm::Ite { cond, then_br, else_br } =>
            has_type(ctx, *cond, Ty::TBool) &&
            has_type(ctx, *then_br, ty) &&
            has_type(ctx, *else_br, ty),
    }
}

// ----------------------------------------------------------------------------
// Canonical Forms Lemmas
// ----------------------------------------------------------------------------

// If a closed value has type Bool, it must be true or false
pub proof fn canonical_forms_bool(t: Tm)
    requires
        value(t),
        has_type(empty_ctx(), t, Ty::TBool),
    ensures t == Tm::Tru || t == Tm::Fls
{
    match t {
        Tm::Tru => {}
        Tm::Fls => {}
        Tm::Abs { .. } => {}  // Cannot have type Bool
        _ => {}  // Not values
    }
}

// If a closed value has arrow type, it must be a lambda
pub proof fn canonical_forms_fun(t: Tm, ty1: Ty, ty2: Ty)
    requires
        value(t),
        has_type(empty_ctx(), t, Ty::TArrow { t1: Box::new(ty1), t2: Box::new(ty2) }),
    ensures exists|x: Var, body: Box<Tm>| t == (Tm::Abs { x, ty: ty1, body })
{
    match t {
        Tm::Abs { x, ty, body } => {
            assert(t == (Tm::Abs { x, ty: ty1, body }));
        }
        Tm::Tru => {}  // Cannot have arrow type
        Tm::Fls => {}  // Cannot have arrow type
        _ => {}
    }
}

// ----------------------------------------------------------------------------
// Progress Theorem
// ----------------------------------------------------------------------------

// A closed, well-typed term is either a value or can step
pub open spec fn progress_holds(t: Tm) -> bool {
    value(t) || step(t).is_some()
}

// Progress for specific Bool terms
pub proof fn progress_true()
    ensures progress_holds(Tm::Tru)
{
    assert(value(Tm::Tru));
}

pub proof fn progress_false()
    ensures progress_holds(Tm::Fls)
{
    assert(value(Tm::Fls));
}

// ----------------------------------------------------------------------------
// Preservation Theorem (Simplified)
// ----------------------------------------------------------------------------

// Type is preserved under reduction
pub open spec fn preservation_holds(t: Tm, ty: Ty) -> bool {
    match step(t) {
        Option::Some(t_prime) => has_type(empty_ctx(), t_prime, ty),
        Option::None => true,
    }
}

// ----------------------------------------------------------------------------
// Stuck Terms
// ----------------------------------------------------------------------------

// A term is stuck if it's not a value and cannot step
pub open spec fn stuck(t: Tm) -> bool {
    !value(t) && step(t).is_none()
}

// Well-typed closed terms are not stuck (specific examples)
pub proof fn well_typed_not_stuck_true()
    ensures !stuck(Tm::Tru)
{
    progress_true();
}

pub proof fn well_typed_not_stuck_false()
    ensures !stuck(Tm::Fls)
{
    progress_false();
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

// true is a value
pub proof fn example_true_value()
    ensures value(Tm::Tru)
{
}

// true has type Bool
pub proof fn example_true_typed()
    ensures has_type(empty_ctx(), Tm::Tru, Ty::TBool)
{
}

// true is not stuck
pub proof fn example_true_not_stuck()
    ensures !stuck(Tm::Tru)
{
}

// if true then false else true --> false
pub proof fn example_if_steps()
{
    let t = Tm::Ite {
        cond: Box::new(Tm::Tru),
        then_br: Box::new(Tm::Fls),
        else_br: Box::new(Tm::Tru)
    };
    assert(step(t) == Option::Some(Tm::Fls));
}

// Lambda is a value
pub proof fn example_abs_value()
{
    let id = Tm::Abs { x: X, ty: Ty::TBool, body: Box::new(Tm::Var { x: X }) };
    assert(value(id));
}

// Application of identity to true steps
pub proof fn example_app_steps()
{
    let id = Tm::Abs { x: X, ty: Ty::TBool, body: Box::new(Tm::Var { x: X }) };
    let app = Tm::App { t1: Box::new(id), t2: Box::new(Tm::Tru) };

    assert(value(Tm::Tru));
    assert(subst(X, Tm::Tru, Tm::Var { x: X }) == Tm::Tru);
    assert(step(app) == Option::Some(Tm::Tru));
}

// Canonical forms: true and false are the only Bool values
pub proof fn example_canonical_bool()
{
    canonical_forms_bool(Tm::Tru);
    canonical_forms_bool(Tm::Fls);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn stlc_prop_examples_verify()
{
    example_true_value();
    example_true_typed();
    example_true_not_stuck();
    example_if_steps();
    example_abs_value();
    example_app_steps();
    example_canonical_bool();

    // Progress examples
    progress_true();
    progress_false();

    // Not stuck examples
    well_typed_not_stuck_true();
    well_typed_not_stuck_false();
}

pub fn main() {
    proof {
        stlc_prop_examples_verify();
    }
}

} // verus!
