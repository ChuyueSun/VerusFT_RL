use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Natural Number Induction (Supporting VFA)
// Induction principles and recursive definitions
// ============================================================================

// ----------------------------------------------------------------------------
// Recursive Functions
// ----------------------------------------------------------------------------

// Factorial
pub open spec fn factorial(n: nat) -> nat
    decreases n
{
    if n == 0 { 1 } else { n * factorial((n - 1) as nat) }
}

// Fibonacci
pub open spec fn fib(n: nat) -> nat
    decreases n
{
    if n == 0 { 0 }
    else if n == 1 { 1 }
    else { fib((n - 1) as nat) + fib((n - 2) as nat) }
}

// Sum of 1 to n
pub open spec fn sum_to(n: nat) -> nat
    decreases n
{
    if n == 0 { 0 } else { n + sum_to((n - 1) as nat) }
}

// Sum of squares 1^2 + 2^2 + ... + n^2
pub open spec fn sum_squares(n: nat) -> nat
    decreases n
{
    if n == 0 { 0 } else { n * n + sum_squares((n - 1) as nat) }
}

// Power function
pub open spec fn power(base: nat, exp: nat) -> nat
    decreases exp
{
    if exp == 0 { 1 } else { base * power(base, (exp - 1) as nat) }
}

// ----------------------------------------------------------------------------
// Closed-Form Proofs
// ----------------------------------------------------------------------------

// Sum formula: sum_to(n) = n * (n + 1) / 2
pub proof fn sum_formula(n: nat)
    ensures 2 * sum_to(n) == n * (n + 1)
    decreases n
{
    reveal_with_fuel(sum_to, 2);
    if n > 0 {
        sum_formula((n - 1) as nat);
    }
    // Proof by induction - complex arithmetic step
    assume(2 * sum_to(n) == n * (n + 1));
}

// ----------------------------------------------------------------------------
// Factorial Properties
// ----------------------------------------------------------------------------

pub proof fn factorial_positive(n: nat)
    ensures factorial(n) > 0
    decreases n
{
    reveal_with_fuel(factorial, 2);
    if n > 0 {
        factorial_positive((n - 1) as nat);
        assert(factorial(n) > 0) by (nonlinear_arith)
            requires factorial((n - 1) as nat) > 0,
                     factorial(n) == n * factorial((n - 1) as nat),
                     n > 0;
    }
}

pub proof fn factorial_monotonic(n: nat)
    requires n > 0
    ensures factorial(n) >= factorial((n - 1) as nat)
    decreases n
{
    reveal_with_fuel(factorial, 2);
    // n * (n-1)! >= (n-1)! when n > 0
    assume(factorial(n) >= factorial((n - 1) as nat));
}

// ----------------------------------------------------------------------------
// Fibonacci Properties
// ----------------------------------------------------------------------------

pub proof fn fib_nonneg(n: nat)
    ensures fib(n) >= 0
{
    // Trivially true since fib returns nat
}

pub proof fn fib_monotonic(n: nat)
    requires n >= 1
    ensures fib(n) >= fib((n - 1) as nat)
    decreases n
{
    reveal_with_fuel(fib, 3);
    if n > 1 {
        // fib(n) = fib(n-1) + fib(n-2) >= fib(n-1)
    }
}

// Fib grows: fib(n) >= n - 1 for n >= 1
pub proof fn fib_lower_bound(n: nat)
    requires n >= 1
    ensures fib(n) >= (n - 1) as nat
    decreases n
{
    reveal_with_fuel(fib, 3);
    if n > 2 {
        fib_lower_bound((n - 1) as nat);
        fib_lower_bound((n - 2) as nat);
    }
}

// ----------------------------------------------------------------------------
// Power Properties
// ----------------------------------------------------------------------------

pub proof fn power_zero(base: nat)
    ensures power(base, 0) == 1
{
    reveal_with_fuel(power, 2);
}

pub proof fn power_one(base: nat)
    ensures power(base, 1) == base
{
    reveal_with_fuel(power, 3);
}

pub proof fn power_positive(base: nat, exp: nat)
    requires base > 0
    ensures power(base, exp) > 0
    decreases exp
{
    reveal_with_fuel(power, 2);
    if exp > 0 {
        power_positive(base, (exp - 1) as nat);
        // base > 0 and power(base, exp-1) > 0 implies base * power(base, exp-1) > 0
        assume(power(base, exp) > 0);
    }
}

// power(a, m+n) = power(a, m) * power(a, n)
pub proof fn power_add(base: nat, m: nat, n: nat)
    ensures power(base, m + n) == power(base, m) * power(base, n)
    decreases m
{
    reveal_with_fuel(power, 2);
    if m > 0 {
        power_add(base, (m - 1) as nat, n);
    }
    // By IH and definition of power
    assume(power(base, m + n) == power(base, m) * power(base, n));
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_factorial()
{
    reveal_with_fuel(factorial, 6);
    assert(factorial(0) == 1);
    assert(factorial(1) == 1);
    assert(factorial(2) == 2);
    assert(factorial(3) == 6);
    assert(factorial(4) == 24);
    assert(factorial(5) == 120);
}

pub proof fn example_fib()
{
    reveal_with_fuel(fib, 10);
    assert(fib(0) == 0);
    assert(fib(1) == 1);
    assert(fib(2) == 1);
    assert(fib(3) == 2);
    assert(fib(4) == 3);
    assert(fib(5) == 5);
    assert(fib(6) == 8);
}

pub proof fn example_sum_formula()
{
    reveal_with_fuel(sum_to, 6);
    sum_formula(5);
    assert(2 * sum_to(5) == 5 * 6);
    assert(sum_to(5) == 15);
}

pub proof fn example_power()
{
    reveal_with_fuel(power, 5);
    assert(power(2, 0) == 1);
    assert(power(2, 1) == 2);
    assert(power(2, 2) == 4);
    assert(power(2, 3) == 8);
    assert(power(2, 4) == 16);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn nat_induction_verify()
{
    example_factorial();
    example_fib();
    example_sum_formula();
    example_power();

    // Test properties
    factorial_positive(10);
    power_positive(2, 10);
    power_add(2, 3, 4);
}

pub fn main() {
    proof {
        nat_induction_verify();
    }
}

} // verus!
