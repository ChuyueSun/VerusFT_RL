use vstd::prelude::*;
use vstd::seq::*;

verus! {

// ============================================================================
// Mutable References (Software Foundations, PLF/References)
// Type system with references, stores, and aliasing
// ============================================================================

// ----------------------------------------------------------------------------
// Types (extended with Unit and Ref)
// ----------------------------------------------------------------------------

pub enum Ty {
    TUnit,                          // Unit type (result of assignment)
    TNat,                           // Natural numbers
    TArrow { t1: Box<Ty>, t2: Box<Ty> },  // Function type
    TRef { t: Box<Ty> },            // Reference type Ref T
}

// ----------------------------------------------------------------------------
// Terms
// ----------------------------------------------------------------------------

pub type Var = nat;
pub type Loc = nat;  // Store locations

pub enum Tm {
    // Lambda calculus core
    Var { x: Var },
    Abs { x: Var, ty: Ty, body: Box<Tm> },
    App { t1: Box<Tm>, t2: Box<Tm> },

    // Natural numbers
    Nat { n: nat },
    Scc { t: Box<Tm> },
    Prd { t: Box<Tm> },

    // Unit
    Unit,

    // References
    Ref { t: Box<Tm> },             // ref t - allocation
    Deref { t: Box<Tm> },           // !t - dereference
    Assign { t1: Box<Tm>, t2: Box<Tm> },  // t1 := t2 - assignment
    Loc { l: Loc },                 // loc l - concrete location
}

// Variable names
pub spec const X: Var = 0;
pub spec const Y: Var = 1;

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

pub open spec fn value(t: Tm) -> bool {
    match t {
        Tm::Abs { .. } => true,
        Tm::Nat { .. } => true,
        Tm::Scc { t } => nvalue(*t),
        Tm::Unit => true,
        Tm::Loc { .. } => true,  // Locations are values
        _ => false,
    }
}

// ----------------------------------------------------------------------------
// Store (as a sequence of terms)
// ----------------------------------------------------------------------------

pub type Store = Seq<Tm>;

pub open spec fn empty_store() -> Store {
    Seq::<Tm>::empty()
}

pub open spec fn store_lookup(st: Store, l: Loc) -> Option<Tm> {
    if l < st.len() {
        Option::Some(st[l as int])
    } else {
        Option::None
    }
}

pub open spec fn store_update(st: Store, l: Loc, v: Tm) -> Store {
    if l < st.len() {
        st.update(l as int, v)
    } else {
        st  // No change if out of bounds
    }
}

pub open spec fn store_extend(st: Store, v: Tm) -> Store {
    st.push(v)
}

// ----------------------------------------------------------------------------
// Store Typing (maps locations to their types)
// ----------------------------------------------------------------------------

pub type StoreTyping = Seq<Ty>;

pub open spec fn empty_store_typing() -> StoreTyping {
    Seq::<Ty>::empty()
}

pub open spec fn store_ty_lookup(st_ty: StoreTyping, l: Loc) -> Option<Ty> {
    if l < st_ty.len() {
        Option::Some(st_ty[l as int])
    } else {
        Option::None
    }
}

pub open spec fn store_ty_extend(st_ty: StoreTyping, ty: Ty) -> StoreTyping {
    st_ty.push(ty)
}

// ----------------------------------------------------------------------------
// Typing Relation (with store typing)
// ----------------------------------------------------------------------------

pub open spec fn has_type(st_ty: StoreTyping, t: Tm, ty: Ty) -> bool
    decreases t
{
    match t {
        // T_Var (simplified - no context for this example)
        Tm::Var { x } => false,  // Variables require context

        // T_Abs
        Tm::Abs { x, ty: ty_param, body } => {
            match ty {
                Ty::TArrow { t1, t2 } => *t1 == ty_param,
                    // Simplified: just check parameter type matches
                _ => false,
            }
        }

        // T_App
        Tm::App { t1, t2 } => false,  // Simplified

        // T_Nat
        Tm::Nat { n } => ty == Ty::TNat,

        // T_Succ
        Tm::Scc { t } => ty == Ty::TNat && has_type(st_ty, *t, Ty::TNat),

        // T_Pred
        Tm::Prd { t } => ty == Ty::TNat && has_type(st_ty, *t, Ty::TNat),

        // T_Unit
        Tm::Unit => ty == Ty::TUnit,

        // T_Loc: A location has type Ref T if the store typing says so
        Tm::Loc { l } => {
            match ty {
                Ty::TRef { t: inner_ty } => {
                    match store_ty_lookup(st_ty, l) {
                        Option::Some(ty_l) => ty_l == *inner_ty,
                        Option::None => false,
                    }
                }
                _ => false,
            }
        }

        // T_Ref: ref t has type Ref T if t has type T
        Tm::Ref { t } => {
            match ty {
                Ty::TRef { t: inner_ty } => has_type(st_ty, *t, *inner_ty),
                _ => false,
            }
        }

        // T_Deref: !t has type T if t has type Ref T
        Tm::Deref { t } => {
            has_type(st_ty, *t, Ty::TRef { t: Box::new(ty) })
        }

        // T_Assign: t1 := t2 has type Unit
        Tm::Assign { t1, t2 } => {
            ty == Ty::TUnit &&
            exists|inner_ty: Ty| #![auto]
                has_type(st_ty, *t1, Ty::TRef { t: Box::new(inner_ty) }) &&
                has_type(st_ty, *t2, inner_ty)
        }
    }
}

// ----------------------------------------------------------------------------
// Store Well-Typedness
// ----------------------------------------------------------------------------

// A store is well-typed if every location contains a value of the expected type
pub open spec fn store_well_typed(st_ty: StoreTyping, st: Store) -> bool {
    st.len() == st_ty.len() &&
    forall|l: nat| #![trigger st[l as int], st_ty[l as int]]
        l < st.len() ==> has_type(st_ty, st[l as int], st_ty[l as int])
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

// Unit has type TUnit
pub proof fn example_unit_type()
    ensures has_type(empty_store_typing(), Tm::Unit, Ty::TUnit)
{
}

// 0 has type TNat
pub proof fn example_nat_type()
    ensures has_type(empty_store_typing(), Tm::Nat { n: 0 }, Ty::TNat)
{
}

// succ(0) has type TNat
pub proof fn example_succ_type()
    ensures has_type(empty_store_typing(), Tm::Scc { t: Box::new(Tm::Nat { n: 0 }) }, Ty::TNat)
{
    let st_ty = empty_store_typing();
    assert(has_type(st_ty, Tm::Nat { n: 0 }, Ty::TNat));
}

// ref 0 has type Ref TNat
pub proof fn example_ref_nat()
    ensures has_type(empty_store_typing(), Tm::Ref { t: Box::new(Tm::Nat { n: 0 }) }, Ty::TRef { t: Box::new(Ty::TNat) })
{
    let st_ty = empty_store_typing();
    assert(has_type(st_ty, Tm::Nat { n: 0 }, Ty::TNat));
}

// Location 0 in store typing [TNat] has type Ref TNat
pub proof fn example_loc_type()
{
    let st_ty = store_ty_extend(empty_store_typing(), Ty::TNat);
    assert(st_ty.len() == 1);
    assert(store_ty_lookup(st_ty, 0) == Option::Some(Ty::TNat));
    assert(has_type(st_ty, Tm::Loc { l: 0 }, Ty::TRef { t: Box::new(Ty::TNat) }));
}

// Dereference: if loc 0 : Ref TNat, then !(loc 0) : TNat
pub proof fn example_deref_type()
{
    let st_ty = store_ty_extend(empty_store_typing(), Ty::TNat);
    assert(has_type(st_ty, Tm::Loc { l: 0 }, Ty::TRef { t: Box::new(Ty::TNat) }));
    assert(has_type(st_ty, Tm::Deref { t: Box::new(Tm::Loc { l: 0 }) }, Ty::TNat));
}

// Values check
pub proof fn example_values()
{
    assert(value(Tm::Unit));
    assert(value(Tm::Nat { n: 42 }));
    assert(value(Tm::Loc { l: 0 }));
    assert(!value(Tm::Ref { t: Box::new(Tm::Nat { n: 0 }) }));  // ref is not a value
    assert(!value(Tm::Deref { t: Box::new(Tm::Loc { l: 0 }) }));  // deref is not a value
}

// ----------------------------------------------------------------------------
// Store Operations
// ----------------------------------------------------------------------------

pub proof fn example_store_ops()
{
    let st = empty_store();
    assert(st.len() == 0);

    // Extend store with a value
    let st1 = store_extend(st, Tm::Nat { n: 42 });
    assert(st1.len() == 1);
    assert(store_lookup(st1, 0) == Option::Some(Tm::Nat { n: 42 }));

    // Update store
    let st2 = store_update(st1, 0, Tm::Nat { n: 100 });
    assert(store_lookup(st2, 0) == Option::Some(Tm::Nat { n: 100 }));
}

pub proof fn example_store_typing_ops()
{
    let st_ty = empty_store_typing();
    assert(st_ty.len() == 0);

    // Extend store typing
    let st_ty1 = store_ty_extend(st_ty, Ty::TNat);
    assert(st_ty1.len() == 1);
    assert(store_ty_lookup(st_ty1, 0) == Option::Some(Ty::TNat));

    // Extend again
    let st_ty2 = store_ty_extend(st_ty1, Ty::TUnit);
    assert(st_ty2.len() == 2);
    assert(store_ty_lookup(st_ty2, 1) == Option::Some(Ty::TUnit));
}

// ----------------------------------------------------------------------------
// Aliasing Example
// ----------------------------------------------------------------------------

// When two locations point to the same cell, updating through one
// affects reads through the other (aliasing)
pub proof fn example_aliasing_concept()
{
    // Create a store with one cell containing 0
    let st = store_extend(empty_store(), Tm::Nat { n: 0 });

    // Location 0 points to this cell
    let loc0 = Tm::Loc { l: 0 };

    // After assignment, the store is updated
    let st_updated = store_update(st, 0, Tm::Nat { n: 42 });

    // Any reference to location 0 now sees 42
    assert(store_lookup(st_updated, 0) == Option::Some(Tm::Nat { n: 42 }));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn references_examples_verify()
{
    example_unit_type();
    example_nat_type();
    example_succ_type();
    example_ref_nat();
    example_loc_type();
    example_deref_type();
    example_values();
    example_store_ops();
    example_store_typing_ops();
    example_aliasing_concept();
}

pub fn main() {
    proof {
        references_examples_verify();
    }
}

} // verus!
