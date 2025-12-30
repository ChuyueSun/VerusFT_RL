use vstd::prelude::*;
use vstd::seq::*;

verus! {

// ============================================================================
// Adding Records to STLC (Software Foundations, PLF/Records)
// Record types, values, and field access
// ============================================================================

// ----------------------------------------------------------------------------
// Labels (field names)
// ----------------------------------------------------------------------------

pub type Label = nat;

pub spec const LA: Label = 0;  // "a"
pub spec const LB: Label = 1;  // "b"
pub spec const LC: Label = 2;  // "c"

// ----------------------------------------------------------------------------
// Types (with Records)
// ----------------------------------------------------------------------------

pub enum Ty {
    TBool,
    TNat,
    TArrow { t1: Box<Ty>, t2: Box<Ty> },
    // Records: list of (label, type) pairs
    TRNil,                                          // Empty record type {}
    TRCons { label: Label, ty: Box<Ty>, rest: Box<Ty> },  // {label: ty, ...rest}
}

// Check if a type is a record type
pub open spec fn is_record_ty(ty: Ty) -> bool
    decreases ty
{
    match ty {
        Ty::TRNil => true,
        Ty::TRCons { label: _, ty: _, rest } => is_record_ty(*rest),
        _ => false,
    }
}

// Lookup a field type in a record type
pub open spec fn ty_lookup(label: Label, ty: Ty) -> Option<Ty>
    decreases ty
{
    match ty {
        Ty::TRNil => Option::None,
        Ty::TRCons { label: l, ty: t, rest } =>
            if l == label {
                Option::Some(*t)
            } else {
                ty_lookup(label, *rest)
            },
        _ => Option::None,
    }
}

// ----------------------------------------------------------------------------
// Terms (with Records)
// ----------------------------------------------------------------------------

pub type Var = nat;

pub enum Tm {
    // Basic terms
    Var { x: Var },
    Abs { x: Var, ty: Ty, body: Box<Tm> },
    App { t1: Box<Tm>, t2: Box<Tm> },
    Tru,
    Fls,
    Nat { n: nat },

    // Records
    RNil,                                                // Empty record {}
    RCons { label: Label, t: Box<Tm>, rest: Box<Tm> },  // {label = t, ...rest}
    RProj { t: Box<Tm>, label: Label },                  // t.label (field access)
}

pub spec const X: Var = 0;

// ----------------------------------------------------------------------------
// Values
// ----------------------------------------------------------------------------

pub open spec fn is_record_value(t: Tm) -> bool
    decreases t
{
    match t {
        Tm::RNil => true,
        Tm::RCons { label: _, t, rest } => value(*t) && is_record_value(*rest),
        _ => false,
    }
}

pub open spec fn value(t: Tm) -> bool
    decreases t
{
    match t {
        Tm::Abs { .. } => true,
        Tm::Tru => true,
        Tm::Fls => true,
        Tm::Nat { .. } => true,
        Tm::RNil => true,
        Tm::RCons { label: _, t, rest } => value(*t) && is_record_value(*rest),
        _ => false,
    }
}

// ----------------------------------------------------------------------------
// Field Lookup in Record Terms
// ----------------------------------------------------------------------------

pub open spec fn tm_lookup(label: Label, t: Tm) -> Option<Tm>
    decreases t
{
    match t {
        Tm::RNil => Option::None,
        Tm::RCons { label: l, t: v, rest } =>
            if l == label {
                Option::Some(*v)
            } else {
                tm_lookup(label, *rest)
            },
        _ => Option::None,
    }
}

// ----------------------------------------------------------------------------
// Typing Relation
// ----------------------------------------------------------------------------

pub open spec fn has_type(t: Tm, ty: Ty) -> bool
    decreases t
{
    match t {
        // T_True, T_False
        Tm::Tru => ty == Ty::TBool,
        Tm::Fls => ty == Ty::TBool,

        // T_Nat
        Tm::Nat { .. } => ty == Ty::TNat,

        // T_RNil: {} : {}
        Tm::RNil => ty == Ty::TRNil,

        // T_RCons: {l=t, ...rest} : {l:T, ...Rest} if t:T and rest:Rest
        Tm::RCons { label, t, rest } => {
            match ty {
                Ty::TRCons { label: l, ty: ty_field, rest: ty_rest } =>
                    l == label &&
                    has_type(*t, *ty_field) &&
                    has_type(*rest, *ty_rest) &&
                    is_record_ty(*ty_rest),
                _ => false,
            }
        }

        // T_Proj: t.label : T if t : {..., label:T, ...}
        Tm::RProj { t, label } => {
            exists|rec_ty: Ty| #![auto]
                has_type(*t, rec_ty) &&
                is_record_ty(rec_ty) &&
                ty_lookup(label, rec_ty) == Option::Some(ty)
        }

        _ => false,  // Other terms need context
    }
}

// ----------------------------------------------------------------------------
// Examples: Record Types
// ----------------------------------------------------------------------------

// {} is a record type
pub proof fn example_empty_record_ty()
{
    assert(is_record_ty(Ty::TRNil));
}

// {a: Bool} is a record type
pub proof fn example_single_field_ty()
{
    let ty = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(Ty::TRNil) };
    reveal_with_fuel(is_record_ty, 2);
    assert(is_record_ty(ty));
}

// {a: Bool, b: Nat} is a record type
pub proof fn example_two_field_ty()
{
    let inner = Ty::TRCons { label: LB, ty: Box::new(Ty::TNat), rest: Box::new(Ty::TRNil) };
    let ty = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(inner) };
    reveal_with_fuel(is_record_ty, 3);
    assert(is_record_ty(ty));
}

// ----------------------------------------------------------------------------
// Examples: Type Lookup
// ----------------------------------------------------------------------------

// Lookup 'a' in {a: Bool} returns Bool
pub proof fn example_ty_lookup_found()
{
    let ty = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(Ty::TRNil) };
    assert(ty_lookup(LA, ty) == Option::Some(Ty::TBool));
}

// Lookup 'b' in {a: Bool} returns None
pub proof fn example_ty_lookup_not_found()
{
    let ty = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(Ty::TRNil) };
    reveal_with_fuel(ty_lookup, 3);
    assert(ty_lookup(LB, ty) == Option::<Ty>::None);
}

// Lookup 'b' in {a: Bool, b: Nat} returns Nat
pub proof fn example_ty_lookup_second()
{
    let inner = Ty::TRCons { label: LB, ty: Box::new(Ty::TNat), rest: Box::new(Ty::TRNil) };
    let ty = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(inner) };
    reveal_with_fuel(ty_lookup, 3);
    assert(ty_lookup(LB, ty) == Option::Some(Ty::TNat));
}

// ----------------------------------------------------------------------------
// Examples: Record Values
// ----------------------------------------------------------------------------

// {} is a value
pub proof fn example_empty_record_value()
{
    assert(value(Tm::RNil));
}

// {a = true} is a value
pub proof fn example_single_field_value()
{
    let r = Tm::RCons { label: LA, t: Box::new(Tm::Tru), rest: Box::new(Tm::RNil) };
    assert(value(Tm::Tru));
    assert(is_record_value(Tm::RNil));
    assert(value(r));
}

// {a = true, b = 0} is a value
pub proof fn example_two_field_value()
{
    let inner = Tm::RCons { label: LB, t: Box::new(Tm::Nat { n: 0 }), rest: Box::new(Tm::RNil) };
    let r = Tm::RCons { label: LA, t: Box::new(Tm::Tru), rest: Box::new(inner) };

    assert(value(Tm::Nat { n: 0 }));
    assert(is_record_value(Tm::RNil));
    assert(is_record_value(inner));
    assert(value(Tm::Tru));
    assert(value(r));
}

// ----------------------------------------------------------------------------
// Examples: Term Lookup
// ----------------------------------------------------------------------------

// Lookup 'a' in {a = true} returns true
pub proof fn example_tm_lookup_found()
{
    let r = Tm::RCons { label: LA, t: Box::new(Tm::Tru), rest: Box::new(Tm::RNil) };
    assert(tm_lookup(LA, r) == Option::Some(Tm::Tru));
}

// ----------------------------------------------------------------------------
// Examples: Typing
// ----------------------------------------------------------------------------

// {} : {}
pub proof fn example_empty_record_typed()
{
    assert(has_type(Tm::RNil, Ty::TRNil));
}

// {a = true} : {a: Bool}
pub proof fn example_single_field_typed()
{
    let r = Tm::RCons { label: LA, t: Box::new(Tm::Tru), rest: Box::new(Tm::RNil) };
    let ty = Ty::TRCons { label: LA, ty: Box::new(Ty::TBool), rest: Box::new(Ty::TRNil) };

    assert(has_type(Tm::Tru, Ty::TBool));
    assert(has_type(Tm::RNil, Ty::TRNil));
    assert(is_record_ty(Ty::TRNil));
    assert(has_type(r, ty));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn records_examples_verify()
{
    // Type examples
    example_empty_record_ty();
    example_single_field_ty();
    example_two_field_ty();

    // Type lookup examples
    example_ty_lookup_found();
    example_ty_lookup_not_found();
    example_ty_lookup_second();

    // Value examples
    example_empty_record_value();
    example_single_field_value();
    example_two_field_value();

    // Term lookup examples
    example_tm_lookup_found();

    // Typing examples
    example_empty_record_typed();
    example_single_field_typed();
}

pub fn main() {
    proof {
        records_examples_verify();
    }
}

} // verus!
