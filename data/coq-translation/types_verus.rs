use vstd::prelude::*;

verus! {

// ============================================================================
// Type Systems (Software Foundations, PLF/Types)
// A typed language with booleans and natural numbers
// ============================================================================

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

pub enum Ty {
    TBool,
    TNat,
}

// ----------------------------------------------------------------------------
// Terms
// ----------------------------------------------------------------------------

pub enum Tm {
    // Booleans
    Tru,
    Fls,
    Ite { t1: Box<Tm>, t2: Box<Tm>, t3: Box<Tm> },  // if-then-else

    // Natural numbers
    Zro,                    // zero
    Scc { t: Box<Tm> },     // successor
    Prd { t: Box<Tm> },     // predecessor
    IsZro { t: Box<Tm> },   // iszero test
}

// ----------------------------------------------------------------------------
// Values
// ----------------------------------------------------------------------------

pub open spec fn bvalue(t: Tm) -> bool {
    match t {
        Tm::Tru => true,
        Tm::Fls => true,
        _ => false,
    }
}

pub open spec fn nvalue(t: Tm) -> bool
    decreases t
{
    match t {
        Tm::Zro => true,
        Tm::Scc { t } => nvalue(*t),
        _ => false,
    }
}

pub open spec fn value(t: Tm) -> bool {
    bvalue(t) || nvalue(t)
}

// ----------------------------------------------------------------------------
// Typing Relation (Simplified)
// ----------------------------------------------------------------------------

pub open spec fn has_type(t: Tm, ty: Ty) -> bool
    decreases t
{
    match t {
        // T_True
        Tm::Tru => ty == Ty::TBool,
        // T_False
        Tm::Fls => ty == Ty::TBool,
        // T_If
        Tm::Ite { t1: cond, t2: then_br, t3: else_br } =>
            has_type(*cond, Ty::TBool) &&
            has_type(*then_br, ty) &&
            has_type(*else_br, ty),
        // T_Zero
        Tm::Zro => ty == Ty::TNat,
        // T_Succ
        Tm::Scc { t } => ty == Ty::TNat && has_type(*t, Ty::TNat),
        // T_Pred
        Tm::Prd { t } => ty == Ty::TNat && has_type(*t, Ty::TNat),
        // T_IsZero
        Tm::IsZro { t } => ty == Ty::TBool && has_type(*t, Ty::TNat),
    }
}

// ----------------------------------------------------------------------------
// Examples: Typing
// ----------------------------------------------------------------------------

// true has type Bool
pub proof fn example_true_bool()
    ensures has_type(Tm::Tru, Ty::TBool)
{
}

// false has type Bool
pub proof fn example_false_bool()
    ensures has_type(Tm::Fls, Ty::TBool)
{
}

// 0 has type Nat
pub proof fn example_zero_nat()
    ensures has_type(Tm::Zro, Ty::TNat)
{
}

// succ(0) has type Nat
pub proof fn example_succ_zero()
{
    assert(has_type(Tm::Zro, Ty::TNat));
    assert(has_type(Tm::Scc { t: Box::new(Tm::Zro) }, Ty::TNat));
}

// pred(0) has type Nat
pub proof fn example_pred_zero()
{
    assert(has_type(Tm::Zro, Ty::TNat));
    assert(has_type(Tm::Prd { t: Box::new(Tm::Zro) }, Ty::TNat));
}

// iszero(0) has type Bool
pub proof fn example_iszero_zero()
{
    assert(has_type(Tm::Zro, Ty::TNat));
    assert(has_type(Tm::IsZro { t: Box::new(Tm::Zro) }, Ty::TBool));
}

// if true then 0 else succ(0) has type Nat
pub proof fn example_if_nat()
{
    assert(has_type(Tm::Tru, Ty::TBool));
    assert(has_type(Tm::Zro, Ty::TNat));
    assert(has_type(Tm::Scc { t: Box::new(Tm::Zro) }, Ty::TNat));
    assert(has_type(Tm::Ite {
        t1: Box::new(Tm::Tru),
        t2: Box::new(Tm::Zro),
        t3: Box::new(Tm::Scc { t: Box::new(Tm::Zro) })
    }, Ty::TNat));
}

// ----------------------------------------------------------------------------
// Examples: Values
// ----------------------------------------------------------------------------

pub proof fn example_values()
{
    // Boolean values
    assert(bvalue(Tm::Tru));
    assert(bvalue(Tm::Fls));
    assert(!bvalue(Tm::Zro));

    // Numeric values
    assert(nvalue(Tm::Zro));
    assert(nvalue(Tm::Scc { t: Box::new(Tm::Zro) }));
    assert(nvalue(Tm::Scc { t: Box::new(Tm::Scc { t: Box::new(Tm::Zro) }) }));

    // Combined
    assert(value(Tm::Tru));
    assert(value(Tm::Zro));
    assert(value(Tm::Scc { t: Box::new(Tm::Zro) }));
}

// ----------------------------------------------------------------------------
// Type Uniqueness
// ----------------------------------------------------------------------------

// A term has at most one type
pub proof fn type_uniqueness(t: Tm, ty1: Ty, ty2: Ty)
    requires
        has_type(t, ty1),
        has_type(t, ty2),
    ensures ty1 == ty2
    decreases t
{
    match t {
        Tm::Tru => {}
        Tm::Fls => {}
        Tm::Zro => {}
        Tm::Scc { t: inner } => {
            // Both must be TNat
        }
        Tm::Prd { t: inner } => {
            // Both must be TNat
        }
        Tm::IsZro { t: inner } => {
            // Both must be TBool
        }
        Tm::Ite { t1: _, t2: then_br, t3: _ } => {
            type_uniqueness(*then_br, ty1, ty2);
        }
    }
}

// ----------------------------------------------------------------------------
// Well-typed values
// ----------------------------------------------------------------------------

// Boolean values have type Bool
pub proof fn bool_value_has_bool_type(t: Tm)
    requires bvalue(t)
    ensures has_type(t, Ty::TBool)
{
    match t {
        Tm::Tru => {}
        Tm::Fls => {}
        _ => {}
    }
}

// Numeric values have type Nat
pub proof fn nat_value_has_nat_type(t: Tm)
    requires nvalue(t)
    ensures has_type(t, Ty::TNat)
    decreases t
{
    match t {
        Tm::Zro => {}
        Tm::Scc { t: inner } => {
            nat_value_has_nat_type(*inner);
        }
        _ => {}
    }
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn types_examples_verify()
{
    example_true_bool();
    example_false_bool();
    example_zero_nat();
    example_succ_zero();
    example_pred_zero();
    example_iszero_zero();
    example_if_nat();
    example_values();

    // Test type uniqueness
    type_uniqueness(Tm::Tru, Ty::TBool, Ty::TBool);
    type_uniqueness(Tm::Zro, Ty::TNat, Ty::TNat);

    // Test value typing
    bool_value_has_bool_type(Tm::Tru);
    nat_value_has_nat_type(Tm::Zro);
    nat_value_has_nat_type(Tm::Scc { t: Box::new(Tm::Zro) });
}

pub fn main() {
    proof {
        types_examples_verify();
    }
}

} // verus!
