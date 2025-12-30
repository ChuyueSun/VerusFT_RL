use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Option Type (Supporting VFA)
// Option type operations and properties
// ============================================================================

// ----------------------------------------------------------------------------
// Option Operations
// ----------------------------------------------------------------------------

pub open spec fn is_some<T>(o: Option<T>) -> bool {
    o.is_some()
}

pub open spec fn is_none<T>(o: Option<T>) -> bool {
    o.is_none()
}

pub open spec fn unwrap_or<T>(o: Option<T>, default: T) -> T {
    match o {
        Some(x) => x,
        None => default,
    }
}

pub open spec fn map_option<T, U>(o: Option<T>, f: spec_fn(T) -> U) -> Option<U> {
    match o {
        Some(x) => Some(f(x)),
        None => None,
    }
}

pub open spec fn and_then<T, U>(o: Option<T>, f: spec_fn(T) -> Option<U>) -> Option<U> {
    match o {
        Some(x) => f(x),
        None => None,
    }
}

pub open spec fn or_else<T>(o: Option<T>, alt: Option<T>) -> Option<T> {
    match o {
        Some(x) => Some(x),
        None => alt,
    }
}

// ----------------------------------------------------------------------------
// Option Properties
// ----------------------------------------------------------------------------

// Some is not None
pub proof fn some_not_none<T>(x: T)
    ensures is_some(Some(x)), !is_none(Some(x))
{
}

// None is not Some
pub proof fn none_is_none<T>()
    ensures is_none::<T>(None), !is_some::<T>(None)
{
}

// Unwrap Some returns value
pub proof fn unwrap_some<T>(x: T, default: T)
    ensures unwrap_or(Some(x), default) == x
{
}

// Unwrap None returns default
pub proof fn unwrap_none<T>(default: T)
    ensures unwrap_or::<T>(None, default) == default
{
}

// Map preserves Some/None
pub proof fn map_some<T, U>(x: T, f: spec_fn(T) -> U)
    ensures is_some(map_option(Some(x), f))
{
}

pub proof fn map_none<T, U>(f: spec_fn(T) -> U)
    ensures is_none(map_option::<T, U>(None, f))
{
}

// ----------------------------------------------------------------------------
// Functor Laws
// ----------------------------------------------------------------------------

// map id = id
pub proof fn map_identity<T>(o: Option<T>)
    ensures map_option(o, |x: T| x) == o
{
}

// map (f . g) = map f . map g
pub proof fn map_compose<T, U, V>(o: Option<T>, f: spec_fn(T) -> U, g: spec_fn(U) -> V)
    ensures map_option(map_option(o, f), g) == map_option(o, |x: T| g(f(x)))
{
}

// ----------------------------------------------------------------------------
// Monad Laws
// ----------------------------------------------------------------------------

// return a >>= f = f a
pub proof fn and_then_return<T, U>(x: T, f: spec_fn(T) -> Option<U>)
    ensures and_then(Some(x), f) == f(x)
{
}

// m >>= return = m
pub proof fn and_then_identity<T>(o: Option<T>)
    ensures and_then(o, |x: T| Some(x)) == o
{
}

// ----------------------------------------------------------------------------
// Or Else Properties
// ----------------------------------------------------------------------------

pub proof fn or_else_some<T>(x: T, alt: Option<T>)
    ensures or_else(Some(x), alt) == Some(x)
{
}

pub proof fn or_else_none<T>(alt: Option<T>)
    ensures or_else::<T>(None, alt) == alt
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_basic()
{
    some_not_none(42nat);
    none_is_none::<nat>();

    unwrap_some(5nat, 0);
    assert(unwrap_or(Some(5nat), 0) == 5);

    unwrap_none::<nat>(0);
    assert(unwrap_or::<nat>(None, 0) == 0);
}

pub proof fn example_map()
{
    let o = Some(5nat);
    let doubled = map_option(o, |x: nat| x * 2);
    map_some(5nat, |x: nat| x * 2);
    assert(is_some(doubled));

    let n: Option<nat> = None;
    let doubled_none = map_option(n, |x: nat| x * 2);
    map_none::<nat, nat>(|x: nat| x * 2);
    assert(is_none(doubled_none));
}

pub proof fn example_and_then()
{
    // Safe division
    let safe_div = |x: nat| -> Option<nat> {
        if x == 0 { None } else { Some((100nat / x) as nat) }
    };

    and_then_return(5nat, safe_div);
    assert(and_then(Some(5nat), safe_div) == safe_div(5));

    assert(and_then::<nat, nat>(None, safe_div).is_none());
}

pub proof fn example_or_else()
{
    or_else_some(10nat, Some(20nat));
    assert(or_else(Some(10nat), Some(20nat)) == Some(10nat));

    or_else_none::<nat>(Some(20nat));
    assert(or_else::<nat>(None, Some(20nat)) == Some(20nat));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn option_def_verify()
{
    example_basic();
    example_map();
    example_and_then();
    example_or_else();

    // Test functor laws
    map_identity(Some(5nat));
    map_identity::<nat>(None);
}

pub fn main() {
    proof {
        option_def_verify();
    }
}

} // verus!
