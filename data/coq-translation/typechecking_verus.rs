use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// Type Checking (Software Foundations, PLF/Typechecking)
// A decidable type checker for STLC
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

pub enum Ty {
    TBool,
    TArrow { t1: Box<Ty>, t2: Box<Ty> },
}

// Type equality (decidable)
pub open spec fn ty_eq(t1: Ty, t2: Ty) -> bool
    decreases t1, t2
{
    match (t1, t2) {
        (Ty::TBool, Ty::TBool) => true,
        (Ty::TArrow { t1: a1, t2: a2 }, Ty::TArrow { t1: b1, t2: b2 }) =>
            ty_eq(*a1, *b1) && ty_eq(*a2, *b2),
        _ => false,
    }
}

// Type equality is reflexive
pub proof fn ty_eq_refl(t: Ty)
    ensures ty_eq(t, t)
    decreases t
{
    match t {
        Ty::TBool => {}
        Ty::TArrow { t1, t2 } => {
            ty_eq_refl(*t1);
            ty_eq_refl(*t2);
        }
    }
}

// Type equality implies structural equality
pub proof fn ty_eq_implies_eq(t1: Ty, t2: Ty)
    requires ty_eq(t1, t2)
    ensures t1 == t2
    decreases t1, t2
{
    match (t1, t2) {
        (Ty::TBool, Ty::TBool) => {}
        (Ty::TArrow { t1: a1, t2: a2 }, Ty::TArrow { t1: b1, t2: b2 }) => {
            ty_eq_implies_eq(*a1, *b1);
            ty_eq_implies_eq(*a2, *b2);
        }
        _ => {}
    }
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
}

pub spec const X: Var = 0;
pub spec const Y: Var = 1;

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

pub open spec fn ctx_lookup(ctx: Context, x: Var) -> Option<Ty> {
    if ctx.dom().contains(x) {
        Option::Some(ctx[x])
    } else {
        Option::None
    }
}

// ----------------------------------------------------------------------------
// Type Checking Algorithm (returns Option<Ty>)
// ----------------------------------------------------------------------------

pub open spec fn type_check(ctx: Context, t: Tm) -> Option<Ty>
    decreases t
{
    match t {
        // TC_Var: lookup variable in context
        Tm::Var { x } => ctx_lookup(ctx, x),

        // TC_Abs: check body with extended context
        Tm::Abs { x, ty, body } => {
            match type_check(ctx_extend(ctx, x, ty), *body) {
                Option::None => Option::None,
                Option::Some(ty_body) => Option::Some(Ty::TArrow { t1: Box::new(ty), t2: Box::new(ty_body) }),
            }
        }

        // TC_App: check function and argument types match
        Tm::App { t1, t2 } => {
            match type_check(ctx, *t1) {
                Option::None => Option::None,
                Option::Some(ty1) => {
                    match ty1 {
                        Ty::TArrow { t1: ty_arg, t2: ty_ret } => {
                            match type_check(ctx, *t2) {
                                Option::None => Option::None,
                                Option::Some(ty2) => {
                                    if ty_eq(*ty_arg, ty2) {
                                        Option::Some(*ty_ret)
                                    } else {
                                        Option::None
                                    }
                                }
                            }
                        }
                        _ => Option::None,  // Not a function type
                    }
                }
            }
        }

        // TC_True, TC_False
        Tm::Tru => Option::Some(Ty::TBool),
        Tm::Fls => Option::Some(Ty::TBool),
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

// true has type Bool
pub proof fn example_true()
    ensures type_check(empty_ctx(), Tm::Tru) == Option::Some(Ty::TBool)
{
}

// false has type Bool
pub proof fn example_false()
    ensures type_check(empty_ctx(), Tm::Fls) == Option::Some(Ty::TBool)
{
}

// Variable in context
pub proof fn example_var_in_ctx()
{
    let ctx = ctx_extend(empty_ctx(), X, Ty::TBool);
    assert(ctx_lookup(ctx, X) == Option::Some(Ty::TBool));
    assert(type_check(ctx, Tm::Var { x: X }) == Option::Some(Ty::TBool));
}

// Variable not in context
pub proof fn example_var_not_in_ctx()
{
    assert(ctx_lookup(empty_ctx(), X) == Option::<Ty>::None);
    assert(type_check(empty_ctx(), Tm::Var { x: X }) == Option::<Ty>::None);
}

// Identity function \x:Bool. x has type Bool -> Bool
pub proof fn example_id_function()
{
    let id = Tm::Abs { x: X, ty: Ty::TBool, body: Box::new(Tm::Var { x: X }) };
    let ctx = empty_ctx();
    let ctx_extended = ctx_extend(ctx, X, Ty::TBool);

    // Body (x) in extended context has type Bool
    assert(ctx_lookup(ctx_extended, X) == Option::Some(Ty::TBool));
    assert(type_check(ctx_extended, Tm::Var { x: X }) == Option::Some(Ty::TBool));

    // So the function has type Bool -> Bool
    assert(type_check(ctx, id) == Option::Some(Ty::TArrow {
        t1: Box::new(Ty::TBool),
        t2: Box::new(Ty::TBool)
    }));
}

// Nested abstraction \x:Bool. \y:Bool. x
pub proof fn example_const_function()
{
    let inner = Tm::Abs { x: Y, ty: Ty::TBool, body: Box::new(Tm::Var { x: X }) };
    let outer = Tm::Abs { x: X, ty: Ty::TBool, body: Box::new(inner) };

    let ctx = empty_ctx();
    let ctx_x = ctx_extend(ctx, X, Ty::TBool);
    let ctx_xy = ctx_extend(ctx_x, Y, Ty::TBool);

    // Inner body (x) has type Bool in ctx_xy
    assert(ctx_lookup(ctx_xy, X) == Option::Some(Ty::TBool));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn typechecking_examples_verify()
{
    example_true();
    example_false();
    example_var_in_ctx();
    example_var_not_in_ctx();
    example_id_function();
    example_const_function();

    // Type equality tests
    ty_eq_refl(Ty::TBool);
    ty_eq_refl(Ty::TArrow { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) });
}

pub fn main() {
    proof {
        typechecking_examples_verify();
    }
}

} // verus!
