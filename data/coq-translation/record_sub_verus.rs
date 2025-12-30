use vstd::prelude::*;

verus! {

// ============================================================================
// Subtyping with Records (Software Foundations, PLF/RecordSub)
// Width, depth, and permutation subtyping for records
// ============================================================================

// ----------------------------------------------------------------------------
// Labels
// ----------------------------------------------------------------------------

pub type Label = nat;

pub spec const LA: Label = 0;
pub spec const LB: Label = 1;
pub spec const LC: Label = 2;

// ----------------------------------------------------------------------------
// Types with Records and Top
// ----------------------------------------------------------------------------

pub enum Ty {
    TTop,                                               // Top type
    TBool,
    TNat,
    TArrow { t1: Box<Ty>, t2: Box<Ty> },
    TRNil,                                              // Empty record {}
    TRCons { label: Label, ty: Box<Ty>, rest: Box<Ty> },
}

// Size for termination
pub open spec fn ty_size(ty: Ty) -> nat
    decreases ty
{
    match ty {
        Ty::TTop => 1,
        Ty::TBool => 1,
        Ty::TNat => 1,
        Ty::TArrow { t1, t2 } => 1 + ty_size(*t1) + ty_size(*t2),
        Ty::TRNil => 1,
        Ty::TRCons { label: _, ty, rest } => 1 + ty_size(*ty) + ty_size(*rest),
    }
}

// Check if type is well-formed record type
pub open spec fn is_record_ty(ty: Ty) -> bool
    decreases ty
{
    match ty {
        Ty::TRNil => true,
        Ty::TRCons { label: _, ty: _, rest } => is_record_ty(*rest),
        _ => false,
    }
}

// ----------------------------------------------------------------------------
// Subtyping Relation
// ----------------------------------------------------------------------------

pub open spec fn subtype(s: Ty, t: Ty) -> bool
    decreases ty_size(s) + ty_size(t)
{
    // S_Top: S <: Top
    if t == Ty::TTop {
        true
    }
    // S_Refl: T <: T
    else if s == t {
        true
    }
    else {
        match (s, t) {
            // S_Arrow: S1 -> S2 <: T1 -> T2 if T1 <: S1 and S2 <: T2
            (Ty::TArrow { t1: s1, t2: s2 }, Ty::TArrow { t1: t1, t2: t2 }) =>
                subtype(*t1, *s1) && subtype(*s2, *t2),

            // S_RcdWidth: {labels...} <: {}
            (Ty::TRCons { .. }, Ty::TRNil) => true,
            (Ty::TRNil, Ty::TRNil) => true,

            // S_RcdDepth: {l:S, ...Sr} <: {l:T, ...Tr} if S <: T and Sr <: Tr
            (Ty::TRCons { label: ls, ty: s_ty, rest: s_rest },
             Ty::TRCons { label: lt, ty: t_ty, rest: t_rest }) =>
                if ls == lt {
                    subtype(*s_ty, *t_ty) && subtype(*s_rest, *t_rest)
                } else {
                    // S_RcdPerm: permutation - look for matching field
                    false  // Simplified: no permutation in this version
                },

            _ => false,
        }
    }
}

// ----------------------------------------------------------------------------
// Subtyping Properties
// ----------------------------------------------------------------------------

// Reflexivity
pub proof fn subtype_refl(ty: Ty)
    ensures subtype(ty, ty)
{
}

// Top is supertype of all
pub proof fn subtype_top(ty: Ty)
    ensures subtype(ty, Ty::TTop)
{
}

// ----------------------------------------------------------------------------
// Examples: Basic Subtyping
// ----------------------------------------------------------------------------

// Bool <: Bool
pub proof fn example_bool_refl()
    ensures subtype(Ty::TBool, Ty::TBool)
{
}

// Bool <: Top
pub proof fn example_bool_top()
    ensures subtype(Ty::TBool, Ty::TTop)
{
}

// Nat <: Top
pub proof fn example_nat_top()
    ensures subtype(Ty::TNat, Ty::TTop)
{
}

// ----------------------------------------------------------------------------
// Examples: Arrow Subtyping
// ----------------------------------------------------------------------------

// (Top -> Bool) <: (Bool -> Bool)
// Because Bool <: Top (contravariant in argument)
pub proof fn example_arrow_contra()
{
    reveal_with_fuel(subtype, 3);
    // T1 <: S1: Bool <: Top (yes)
    // S2 <: T2: Bool <: Bool (yes)
    assert(subtype(Ty::TBool, Ty::TTop));
    assert(subtype(Ty::TBool, Ty::TBool));
    assert(subtype(
        Ty::TArrow { t1: Box::new(Ty::TTop), t2: Box::new(Ty::TBool) },
        Ty::TArrow { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) }
    ));
}

// (Bool -> Bool) <: (Bool -> Top)
// Because Bool <: Top (covariant in return)
pub proof fn example_arrow_co()
{
    reveal_with_fuel(subtype, 3);
    assert(subtype(Ty::TBool, Ty::TBool));
    assert(subtype(Ty::TBool, Ty::TTop));
    assert(subtype(
        Ty::TArrow { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TBool) },
        Ty::TArrow { t1: Box::new(Ty::TBool), t2: Box::new(Ty::TTop) }
    ));
}

// ----------------------------------------------------------------------------
// Examples: Record Width Subtyping
// ----------------------------------------------------------------------------

// {a: Bool} <: {}
pub proof fn example_width_single()
{
    let s = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(Ty::TRNil) };
    let t = Ty::TRNil;
    assert(subtype(s, t));
}

// {a: Bool, b: Nat} <: {}
pub proof fn example_width_double()
{
    let inner = Ty::TRCons { label: LB, ty: Box::new(Ty::TNat), rest: Box::new(Ty::TRNil) };
    let s = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(inner) };
    let t = Ty::TRNil;
    assert(subtype(s, t));
}

// ----------------------------------------------------------------------------
// Examples: Record Depth Subtyping
// ----------------------------------------------------------------------------

// {a: Bool} <: {a: Bool}
pub proof fn example_depth_refl()
{
    let ty = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(Ty::TRNil) };
    assert(subtype(ty, ty));
}

// {a: Bool} <: {a: Top}
pub proof fn example_depth_top()
{
    let s = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(Ty::TRNil) };
    let t = Ty::TRCons { label: LA, ty: Box::new(Ty::TTop), rest: Box::new(Ty::TRNil) };
    reveal_with_fuel(subtype, 3);
    assert(subtype(Ty::TBool, Ty::TTop));
    assert(subtype(Ty::TRNil, Ty::TRNil));
    assert(subtype(s, t));
}

// {a: Bool, b: Nat} <: {a: Top, b: Nat}
pub proof fn example_depth_first()
{
    let s_inner = Ty::TRCons { label: LB, ty: Box::new(Ty::TNat), rest: Box::new(Ty::TRNil) };
    let s = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(s_inner) };

    let t_inner = Ty::TRCons { label: LB, ty: Box::new(Ty::TNat), rest: Box::new(Ty::TRNil) };
    let t = Ty::TRCons { label: LA, ty: Box::new(Ty::TTop), rest: Box::new(t_inner) };

    reveal_with_fuel(subtype, 5);
    assert(subtype(Ty::TBool, Ty::TTop));
    assert(subtype(Ty::TNat, Ty::TNat));
    assert(subtype(Ty::TRNil, Ty::TRNil));
    assert(subtype(s_inner, t_inner));
    assert(subtype(s, t));
}

// ----------------------------------------------------------------------------
// Examples: Combined Width and Depth
// ----------------------------------------------------------------------------

// {a: Bool, b: Nat} <: {a: Top}
pub proof fn example_width_and_depth()
{
    let s_inner = Ty::TRCons { label: LB, ty: Box::new(Ty::TNat), rest: Box::new(Ty::TRNil) };
    let s = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(s_inner) };

    let t = Ty::TRCons { label: LA, ty: Box::new(Ty::TTop), rest: Box::new(Ty::TRNil) };

    // This combines width (dropping b) and depth (Bool <: Top)
    // In our simplified version, we need the "rest" to also be subtypes
    // {b: Nat} <: {} (width)
    reveal_with_fuel(subtype, 4);
    assert(subtype(Ty::TBool, Ty::TTop));
    assert(subtype(s_inner, Ty::TRNil));
    assert(subtype(s, t));
}

// ----------------------------------------------------------------------------
// Examples: Negative Cases
// ----------------------------------------------------------------------------

// NOT: {} <: {a: Bool}
pub proof fn example_not_width_reverse()
    ensures !subtype(Ty::TRNil, Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(Ty::TRNil) })
{
}

// NOT: {a: Top} <: {a: Bool}
pub proof fn example_not_depth_reverse()
{
    reveal_with_fuel(subtype, 3);
    // Top is not a subtype of Bool (only Top <: Top and Bool <: Bool)
    assert(!subtype(Ty::TTop, Ty::TBool));
    assert(!subtype(
        Ty::TRCons { label: LA, ty: Box::new(Ty::TTop), rest: Box::new(Ty::TRNil) },
        Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(Ty::TRNil) }
    ));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn record_sub_examples_verify()
{
    // Basic subtyping
    example_bool_refl();
    example_bool_top();
    example_nat_top();
    subtype_refl(Ty::TBool);
    subtype_top(Ty::TNat);

    // Arrow subtyping
    example_arrow_contra();
    example_arrow_co();

    // Record width subtyping
    example_width_single();
    example_width_double();

    // Record depth subtyping
    example_depth_refl();
    example_depth_top();
    example_depth_first();

    // Combined
    example_width_and_depth();

    // Negative cases
    example_not_width_reverse();
    example_not_depth_reverse();
}

pub fn main() {
    proof {
        record_sub_examples_verify();
    }
}

} // verus!
