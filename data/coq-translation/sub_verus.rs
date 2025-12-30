use vstd::prelude::*;

verus! {

// ============================================================================
// Subtyping (Software Foundations, PLF/Sub)
// ============================================================================

// ----------------------------------------------------------------------------
// Types with Subtyping
// ----------------------------------------------------------------------------

pub enum Ty {
    TTop,                                    // Top type (supertype of all)
    TBool,
    TArrow { t1: Box<Ty>, t2: Box<Ty> },    // Function type
    TProd { t1: Box<Ty>, t2: Box<Ty> },     // Product type (pairs)
}

// Size of a type (for termination)
pub open spec fn ty_size(t: Ty) -> nat
    decreases t
{
    match t {
        Ty::TTop => 1,
        Ty::TBool => 1,
        Ty::TArrow { t1, t2 } => 1 + ty_size(*t1) + ty_size(*t2),
        Ty::TProd { t1, t2 } => 1 + ty_size(*t1) + ty_size(*t2),
    }
}

// ----------------------------------------------------------------------------
// Subtype Relation: S <: T (simplified without arrow contravariance)
// ----------------------------------------------------------------------------

pub open spec fn subtype(s: Ty, t: Ty) -> bool
    decreases ty_size(s) + ty_size(t)
{
    // S_Top: Any type is a subtype of Top
    if t == Ty::TTop {
        true
    }
    // S_Refl for base types
    else if s == t {
        true
    }
    // S_Prod: Covariant in both components
    else {
        match (s, t) {
            (Ty::TProd { t1: s1, t2: s2 }, Ty::TProd { t1: t1, t2: t2 }) =>
                subtype(*s1, *t1) && subtype(*s2, *t2),
            _ => false,
        }
    }
}

// ----------------------------------------------------------------------------
// Subtyping Properties
// ----------------------------------------------------------------------------

// Reflexivity: T <: T
pub proof fn subtype_refl(t: Ty)
    ensures subtype(t, t)
{
}

// Top is a supertype of everything
pub proof fn subtype_top(t: Ty)
    ensures subtype(t, Ty::TTop)
{
}

// Bool is only a subtype of Bool and Top
pub proof fn subtype_bool_cases(s: Ty)
    requires subtype(s, Ty::TBool)
    ensures s == Ty::TBool
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

// Bool <: Bool
pub proof fn example_bool_subtype_bool()
    ensures subtype(Ty::TBool, Ty::TBool)
{
}

// Bool <: Top
pub proof fn example_bool_subtype_top()
    ensures subtype(Ty::TBool, Ty::TTop)
{
}

// (Bool -> Bool) <: Top
pub proof fn example_arrow_subtype_top()
    ensures subtype(
        Ty::TArrow { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) },
        Ty::TTop
    )
{
}

// (Bool * Bool) <: (Bool * Bool)
pub proof fn example_prod_refl()
    ensures subtype(
        Ty::TProd { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) },
        Ty::TProd { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) }
    )
{
}

// (Bool * Bool) <: Top
pub proof fn example_prod_subtype_top()
    ensures subtype(
        Ty::TProd { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) },
        Ty::TTop
    )
{
}

// NOT: Bool <: (Bool -> Bool)
pub proof fn example_not_bool_subtype_arrow()
    ensures !subtype(Ty::TBool, Ty::TArrow { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) })
{
}

// NOT: (Bool -> Bool) <: Bool
pub proof fn example_not_arrow_subtype_bool()
    ensures !subtype(Ty::TArrow { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) }, Ty::TBool)
{
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn sub_examples_verify()
{
    // Test reflexivity
    subtype_refl(Ty::TBool);
    subtype_refl(Ty::TTop);
    subtype_refl(Ty::TArrow { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) });

    // Test top as supertype
    subtype_top(Ty::TBool);
    subtype_top(Ty::TArrow { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) });

    // Test examples
    example_bool_subtype_bool();
    example_bool_subtype_top();
    example_arrow_subtype_top();
    example_prod_refl();
    example_prod_subtype_top();
    example_not_bool_subtype_arrow();
    example_not_arrow_subtype_bool();
}

pub fn main() {
    proof {
        sub_examples_verify();
    }
}

} // verus!
