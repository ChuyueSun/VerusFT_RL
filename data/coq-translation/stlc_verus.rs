use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// Simply Typed Lambda Calculus (Software Foundations, PLF/Stlc)
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

pub enum Ty {
    TBool,
    TArrow { t1: Box<Ty>, t2: Box<Ty> },  // Function type T1 -> T2
}

// ----------------------------------------------------------------------------
// Terms
// ----------------------------------------------------------------------------

pub type Var = nat;

pub enum Tm {
    Var { x: Var },                                    // Variable
    Abs { x: Var, ty: Ty, body: Box<Tm> },            // Lambda abstraction \x:T.t
    App { t1: Box<Tm>, t2: Box<Tm> },                 // Application t1 t2
    Tru,                                               // true
    Fls,                                               // false
    Ite { cond: Box<Tm>, then_br: Box<Tm>, else_br: Box<Tm> },  // if-then-else
}

// Variable names for examples
pub spec const X: Var = 0;
pub spec const Y: Var = 1;
pub spec const Z: Var = 2;

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
// Free Variables
// ----------------------------------------------------------------------------

pub open spec fn free_in(x: Var, t: Tm) -> bool
    decreases t
{
    match t {
        Tm::Var { x: y } => x == y,
        Tm::Abs { x: y, ty: _, body } => x != y && free_in(x, *body),
        Tm::App { t1, t2 } => free_in(x, *t1) || free_in(x, *t2),
        Tm::Tru => false,
        Tm::Fls => false,
        Tm::Ite { cond, then_br, else_br } =>
            free_in(x, *cond) || free_in(x, *then_br) || free_in(x, *else_br),
    }
}

// Closed term: no free variables
pub open spec fn closed(t: Tm) -> bool {
    forall|x: Var| !free_in(x, t)
}

// ----------------------------------------------------------------------------
// Substitution [x := s]t
// ----------------------------------------------------------------------------

pub open spec fn subst(x: Var, s: Tm, t: Tm) -> Tm
    decreases t
{
    match t {
        Tm::Var { x: y } => if x == y { s } else { t },
        Tm::Abs { x: y, ty, body } =>
            if x == y {
                t  // x is bound, no substitution in body
            } else {
                Tm::Abs { x: y, ty, body: Box::new(subst(x, s, *body)) }
            },
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
// Small-step Semantics (Call-by-Value)
// ----------------------------------------------------------------------------

pub open spec fn step(t1: Tm, t2: Tm) -> bool
    decreases t1
{
    match t1 {
        // ST_AppAbs: (\x:T.t) v --> [x:=v]t (beta reduction)
        Tm::App { t1: func, t2: arg } => {
            match *func {
                Tm::Abs { x, ty: _, body } if value(*arg) =>
                    t2 == subst(x, *arg, *body),
                _ => {
                    // ST_App1: t1 --> t1' => t1 t2 --> t1' t2
                    if !value(*func) {
                        exists|func_prime: Tm| step(*func, func_prime) &&
                            t2 == Tm::App { t1: Box::new(func_prime), t2: arg }
                    }
                    // ST_App2: t2 --> t2' => v1 t2 --> v1 t2'
                    else if value(*func) && !value(*arg) {
                        exists|arg_prime: Tm| step(*arg, arg_prime) &&
                            t2 == Tm::App { t1: func, t2: Box::new(arg_prime) }
                    }
                    else {
                        false
                    }
                }
            }
        }
        // ST_IfTrue
        Tm::Ite { cond, then_br, else_br } => {
            match *cond {
                Tm::Tru => t2 == *then_br,
                Tm::Fls => t2 == *else_br,
                // ST_If: cond --> cond'
                _ => exists|cond_prime: Tm| step(*cond, cond_prime) &&
                    t2 == Tm::Ite { cond: Box::new(cond_prime), then_br, else_br }
            }
        }
        _ => false,
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
// Typing Relation: ctx |- t : T
// ----------------------------------------------------------------------------

pub open spec fn has_type(ctx: Context, t: Tm, ty: Ty) -> bool
    decreases t
{
    match t {
        // T_Var
        Tm::Var { x } => ctx.dom().contains(x) && ctx[x] == ty,

        // T_Abs
        Tm::Abs { x, ty: ty_param, body } => {
            match ty {
                Ty::TArrow { t1, t2 } =>
                    *t1 == ty_param &&
                    has_type(ctx_extend(ctx, x, ty_param), *body, *t2),
                _ => false,
            }
        }

        // T_App
        Tm::App { t1, t2 } => {
            exists|ty_arg: Ty|
                has_type(ctx, *t1, Ty::TArrow { t1: Box::new(ty_arg), t2: Box::new(ty) }) &&
                has_type(ctx, *t2, ty_arg)
        }

        // T_True
        Tm::Tru => ty == Ty::TBool,

        // T_False
        Tm::Fls => ty == Ty::TBool,

        // T_If
        Tm::Ite { cond, then_br, else_br } =>
            has_type(ctx, *cond, Ty::TBool) &&
            has_type(ctx, *then_br, ty) &&
            has_type(ctx, *else_br, ty),
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

// Identity function: \x:Bool. x
pub open spec fn id_bool() -> Tm {
    Tm::Abs { x: X, ty: Ty::TBool, body: Box::new(Tm::Var { x: X }) }
}

// Identity function: variable in context has correct type
pub proof fn example_var_in_context()
{
    let ctx = ctx_extend(empty_ctx(), X, Ty::TBool);
    assert(ctx.dom().contains(X));
    assert(ctx[X] == Ty::TBool);
    assert(has_type(ctx, Tm::Var { x: X }, Ty::TBool));
}

// Constant function: \x:Bool. \y:Bool. x
pub open spec fn const_bool() -> Tm {
    Tm::Abs {
        x: X,
        ty: Ty::TBool,
        body: Box::new(Tm::Abs {
            x: Y,
            ty: Ty::TBool,
            body: Box::new(Tm::Var { x: X })
        })
    }
}

// true is a value
pub proof fn example_true_value()
    ensures value(Tm::Tru)
{
}

// Lambda is a value
pub proof fn example_abs_value()
    ensures value(id_bool())
{
}

// Substitution example: [x:=true](x) = true
pub proof fn example_subst_var()
    ensures subst(X, Tm::Tru, Tm::Var { x: X }) == Tm::Tru
{
}

// Substitution example: [x:=true](y) = y
pub proof fn example_subst_other_var()
    ensures subst(X, Tm::Tru, Tm::Var { x: Y }) == (Tm::Var { x: Y })
{
}

// Substitution doesn't go under same-named binder
pub proof fn example_subst_shadow()
    ensures subst(X, Tm::Tru, Tm::Abs { x: X, ty: Ty::TBool, body: Box::new(Tm::Var { x: X }) })
        == (Tm::Abs { x: X, ty: Ty::TBool, body: Box::new(Tm::Var { x: X }) })
{
}

// Beta reduction: (\x:Bool. x) true --> true
pub proof fn example_beta_reduction()
    ensures step(
        Tm::App { t1: Box::new(id_bool()), t2: Box::new(Tm::Tru) },
        Tm::Tru
    )
{
    assert(value(Tm::Tru));
    assert(subst(X, Tm::Tru, Tm::Var { x: X }) == Tm::Tru);
}

// if true then false else true --> false
pub proof fn example_if_true()
    ensures step(
        Tm::Ite { cond: Box::new(Tm::Tru), then_br: Box::new(Tm::Fls), else_br: Box::new(Tm::Tru) },
        Tm::Fls
    )
{
}

// Free variable check
pub proof fn example_free_var()
{
    assert(free_in(X, Tm::Var { x: X }));
    assert(!free_in(Y, Tm::Var { x: X }));
    assert(!free_in(X, Tm::Abs { x: X, ty: Ty::TBool, body: Box::new(Tm::Var { x: X }) }));
    assert(free_in(X, Tm::Abs { x: Y, ty: Ty::TBool, body: Box::new(Tm::Var { x: X }) }));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn stlc_examples_verify()
{
    example_var_in_context();
    example_true_value();
    example_abs_value();
    example_subst_var();
    example_subst_other_var();
    example_subst_shadow();
    example_beta_reduction();
    example_if_true();
    example_free_var();
}

pub fn main() {
    proof {
        stlc_examples_verify();
    }
}

} // verus!
