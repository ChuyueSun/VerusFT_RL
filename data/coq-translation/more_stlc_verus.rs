use vstd::prelude::*;

verus! {

// ============================================================================
// More on the Simply Typed Lambda-Calculus (Software Foundations, PLF/MoreStlc)
// Extensions: let, pairs, sums, lists, fix
// ============================================================================

// ----------------------------------------------------------------------------
// Extended Types
// ----------------------------------------------------------------------------

pub enum Ty {
    TUnit,                              // Unit type
    TBool,                              // Booleans
    TNat,                               // Natural numbers
    TArrow { t1: Box<Ty>, t2: Box<Ty> },  // Functions
    TProd { t1: Box<Ty>, t2: Box<Ty> },   // Products (pairs)
    TSum { t1: Box<Ty>, t2: Box<Ty> },    // Sums (either)
    TList { t: Box<Ty> },                 // Lists
}

// ----------------------------------------------------------------------------
// Extended Terms
// ----------------------------------------------------------------------------

pub type Var = nat;

pub enum Tm {
    // Basic lambda calculus
    Var { x: Var },
    Abs { x: Var, ty: Ty, body: Box<Tm> },
    App { t1: Box<Tm>, t2: Box<Tm> },

    // Unit
    Unit,

    // Booleans
    Tru,
    Fls,
    Ite { cond: Box<Tm>, then_br: Box<Tm>, else_br: Box<Tm> },

    // Natural numbers
    Nat { n: nat },
    Scc { t: Box<Tm> },
    Prd { t: Box<Tm> },
    IsZro { t: Box<Tm> },

    // Let binding
    Let { x: Var, t1: Box<Tm>, t2: Box<Tm> },  // let x = t1 in t2

    // Pairs (products)
    Pair { t1: Box<Tm>, t2: Box<Tm> },  // (t1, t2)
    Fst { t: Box<Tm> },                  // t.fst
    Snd { t: Box<Tm> },                  // t.snd

    // Sums
    Inl { t: Box<Tm>, ty: Ty },          // inl t as T
    Inr { t: Box<Tm>, ty: Ty },          // inr t as T
    Case { t: Box<Tm>, x1: Var, t1: Box<Tm>, x2: Var, t2: Box<Tm> },

    // Lists
    Nil { ty: Ty },                       // nil T
    Cons { t1: Box<Tm>, t2: Box<Tm> },   // cons t1 t2
    LCase { t: Box<Tm>, t1: Box<Tm>, x1: Var, x2: Var, t2: Box<Tm> },

    // Fixed point (recursion)
    Fix { t: Box<Tm> },
}

pub spec const X: Var = 0;
pub spec const Y: Var = 1;
pub spec const Z: Var = 2;

// ----------------------------------------------------------------------------
// Values
// ----------------------------------------------------------------------------

pub open spec fn nvalue(t: Tm) -> bool
    decreases t
{
    match t {
        Tm::Nat { .. } => true,
        Tm::Scc { t } => nvalue(*t),
        _ => false,
    }
}

pub open spec fn lvalue(t: Tm) -> bool
    decreases t
{
    match t {
        Tm::Nil { .. } => true,
        Tm::Cons { t1, t2 } => value(*t1) && lvalue(*t2),
        _ => false,
    }
}

pub open spec fn value(t: Tm) -> bool
    decreases t
{
    match t {
        Tm::Unit => true,
        Tm::Abs { .. } => true,
        Tm::Tru => true,
        Tm::Fls => true,
        Tm::Nat { .. } => true,
        Tm::Scc { t } => nvalue(*t),
        Tm::Pair { t1, t2 } => value(*t1) && value(*t2),
        Tm::Inl { t, .. } => value(*t),
        Tm::Inr { t, .. } => value(*t),
        Tm::Nil { .. } => true,
        Tm::Cons { t1, t2 } => value(*t1) && lvalue(*t2),
        _ => false,
    }
}

// ----------------------------------------------------------------------------
// Typing Relation (Simplified - just showing key rules)
// ----------------------------------------------------------------------------

pub open spec fn has_type_simple(t: Tm, ty: Ty) -> bool
    decreases t
{
    match t {
        // Unit
        Tm::Unit => ty == Ty::TUnit,

        // Booleans
        Tm::Tru => ty == Ty::TBool,
        Tm::Fls => ty == Ty::TBool,

        // Naturals
        Tm::Nat { .. } => ty == Ty::TNat,
        Tm::Scc { t } => ty == Ty::TNat && has_type_simple(*t, Ty::TNat),
        Tm::Prd { t } => ty == Ty::TNat && has_type_simple(*t, Ty::TNat),
        Tm::IsZro { t } => ty == Ty::TBool && has_type_simple(*t, Ty::TNat),

        // Pairs: (t1, t2) : T1 * T2
        Tm::Pair { t1, t2 } => {
            match ty {
                Ty::TProd { t1: ty1, t2: ty2 } =>
                    has_type_simple(*t1, *ty1) && has_type_simple(*t2, *ty2),
                _ => false,
            }
        }

        // Fst: t.fst : T1 if t : T1 * T2
        Tm::Fst { t } => {
            exists|ty2: Ty| #![auto]
                has_type_simple(*t, Ty::TProd { t1: Box::new(ty), t2: Box::new(ty2) })
        }

        // Snd: t.snd : T2 if t : T1 * T2
        Tm::Snd { t } => {
            exists|ty1: Ty| #![auto]
                has_type_simple(*t, Ty::TProd { t1: Box::new(ty1), t2: Box::new(ty) })
        }

        // Inl: inl t as T1+T2 : T1+T2 if t : T1
        Tm::Inl { t, ty: ann_ty } => {
            ty == ann_ty &&
            match ann_ty {
                Ty::TSum { t1, t2: _ } => has_type_simple(*t, *t1),
                _ => false,
            }
        }

        // Inr: inr t as T1+T2 : T1+T2 if t : T2
        Tm::Inr { t, ty: ann_ty } => {
            ty == ann_ty &&
            match ann_ty {
                Ty::TSum { t1: _, t2 } => has_type_simple(*t, *t2),
                _ => false,
            }
        }

        // Nil: nil T : List T
        Tm::Nil { ty: elem_ty } => ty == (Ty::TList { t: Box::new(elem_ty) }),

        _ => false,  // Other cases need context
    }
}

// ----------------------------------------------------------------------------
// Examples: Let Bindings
// ----------------------------------------------------------------------------

// let x = 5 in x + 1 (conceptually evaluates to 6)
pub proof fn example_let_concept()
{
    // let x = 5 in succ(x)
    let t = Tm::Let {
        x: X,
        t1: Box::new(Tm::Nat { n: 5 }),
        t2: Box::new(Tm::Scc { t: Box::new(Tm::Var { x: X }) })
    };
    // This represents a let binding
    assert(!value(t));
}

// ----------------------------------------------------------------------------
// Examples: Pairs
// ----------------------------------------------------------------------------

// (true, 0) is a value
pub proof fn example_pair_value()
{
    let p = Tm::Pair { t1: Box::new(Tm::Tru), t2: Box::new(Tm::Nat { n: 0 }) };
    assert(value(Tm::Tru));
    assert(value(Tm::Nat { n: 0 }));
    assert(value(p));
}

// (true, 0) : Bool * Nat
pub proof fn example_pair_type()
{
    let p = Tm::Pair { t1: Box::new(Tm::Tru), t2: Box::new(Tm::Nat { n: 0 }) };
    let ty = Ty::TProd { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TNat) };

    assert(has_type_simple(Tm::Tru, Ty::TBool));
    assert(has_type_simple(Tm::Nat { n: 0 }, Ty::TNat));
    assert(has_type_simple(p, ty));
}

// ----------------------------------------------------------------------------
// Examples: Sums
// ----------------------------------------------------------------------------

// inl true as Bool+Nat is a value
pub proof fn example_inl_value()
{
    let sum_ty = Ty::TSum { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TNat) };
    let t = Tm::Inl { t: Box::new(Tm::Tru), ty: sum_ty };

    assert(value(Tm::Tru));
    assert(value(t));
}

// inr 0 as Bool+Nat is a value
pub proof fn example_inr_value()
{
    let sum_ty = Ty::TSum { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TNat) };
    let t = Tm::Inr { t: Box::new(Tm::Nat { n: 0 }), ty: sum_ty };

    assert(value(Tm::Nat { n: 0 }));
    assert(value(t));
}

// inl true as Bool+Nat : Bool+Nat
pub proof fn example_inl_type()
{
    let sum_ty = Ty::TSum { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TNat) };
    let t = Tm::Inl { t: Box::new(Tm::Tru), ty: sum_ty };

    assert(has_type_simple(Tm::Tru, Ty::TBool));
    assert(has_type_simple(t, sum_ty));
}

// ----------------------------------------------------------------------------
// Examples: Lists
// ----------------------------------------------------------------------------

// nil Nat is a value
pub proof fn example_nil_value()
{
    let t = Tm::Nil { ty: Ty::TNat };
    assert(value(t));
}

// nil Nat : List Nat
pub proof fn example_nil_type()
{
    let t = Tm::Nil { ty: Ty::TNat };
    let ty = Ty::TList { t: Box::new(Ty::TNat) };
    assert(has_type_simple(t, ty));
}

// cons 1 (nil Nat) is a value
pub proof fn example_cons_value()
{
    let nil = Tm::Nil { ty: Ty::TNat };
    let t = Tm::Cons { t1: Box::new(Tm::Nat { n: 1 }), t2: Box::new(nil) };

    assert(value(Tm::Nat { n: 1 }));
    assert(lvalue(Tm::Nil { ty: Ty::TNat }));
    assert(value(t));
}

// ----------------------------------------------------------------------------
// Examples: Fixed Point
// ----------------------------------------------------------------------------

// fix (\f:Nat->Nat. \x:Nat. ...) represents a recursive function
pub proof fn example_fix_concept()
{
    // fix is used for recursion
    // fix F reduces to F (fix F)
    let f_ty = Ty::TArrow { t1: Box::new(Ty::TNat), t2: Box::new(Ty::TNat) };
    let body = Tm::Abs { x: X, ty: Ty::TNat, body: Box::new(Tm::Var { x: X }) };
    let f = Tm::Abs { x: Y, ty: f_ty, body: Box::new(body) };
    let fix_f = Tm::Fix { t: Box::new(f) };

    // fix is not a value
    assert(!value(fix_f));
}

// ----------------------------------------------------------------------------
// Examples: Unit
// ----------------------------------------------------------------------------

pub proof fn example_unit()
{
    assert(value(Tm::Unit));
    assert(has_type_simple(Tm::Unit, Ty::TUnit));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn more_stlc_examples_verify()
{
    example_let_concept();
    example_pair_value();
    example_pair_type();
    example_inl_value();
    example_inr_value();
    example_inl_type();
    example_nil_value();
    example_nil_type();
    example_cons_value();
    example_fix_concept();
    example_unit();
}

pub fn main() {
    proof {
        more_stlc_examples_verify();
    }
}

} // verus!
