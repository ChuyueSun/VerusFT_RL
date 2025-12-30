use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Result Type (Supporting VFA)
// Result type for error handling
// ============================================================================

pub enum Result<T, E> {
    Ok(T),
    Err(E),
}

pub open spec fn is_ok<T, E>(r: Result<T, E>) -> bool { matches!(r, Result::Ok(_)) }
pub open spec fn is_err<T, E>(r: Result<T, E>) -> bool { matches!(r, Result::Err(_)) }

pub open spec fn unwrap_ok<T, E>(r: Result<T, E>) -> T
    recommends is_ok(r)
{
    match r { Result::Ok(t) => t, Result::Err(_) => arbitrary() }
}

pub open spec fn unwrap_err<T, E>(r: Result<T, E>) -> E
    recommends is_err(r)
{
    match r { Result::Ok(_) => arbitrary(), Result::Err(e) => e }
}

pub open spec fn map_ok<T, U, E>(r: Result<T, E>, f: spec_fn(T) -> U) -> Result<U, E> {
    match r {
        Result::Ok(t) => Result::Ok(f(t)),
        Result::Err(e) => Result::Err(e),
    }
}

pub open spec fn map_err<T, E, F>(r: Result<T, E>, f: spec_fn(E) -> F) -> Result<T, F> {
    match r {
        Result::Ok(t) => Result::Ok(t),
        Result::Err(e) => Result::Err(f(e)),
    }
}

pub proof fn ok_not_err<T, E>(t: T)
    ensures is_ok(Result::<T, E>::Ok(t)), !is_err(Result::<T, E>::Ok(t))
{}

pub proof fn err_not_ok<T, E>(e: E)
    ensures is_err(Result::<T, E>::Err(e)), !is_ok(Result::<T, E>::Err(e))
{}

pub proof fn example_result()
{
    let r: Result<nat, bool> = Result::Ok(42);
    ok_not_err::<nat, bool>(42);
    assert(is_ok(r));
}

pub proof fn result_def_verify()
{
    example_result();
}

pub fn main() { proof { result_def_verify(); } }

} // verus!
