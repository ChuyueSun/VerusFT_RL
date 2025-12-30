use vstd::prelude::*;

verus! {

// ============================================================================
// Small-step Operational Semantics (Software Foundations, PLF/Smallstep)
// ============================================================================

// ----------------------------------------------------------------------------
// A Simple Language of Arithmetic Expressions
// ----------------------------------------------------------------------------

pub enum Tm {
    C { n: int },              // Constant
    P { t1: Box<Tm>, t2: Box<Tm> },  // Plus
}

// ----------------------------------------------------------------------------
// Values
// ----------------------------------------------------------------------------

pub open spec fn value(t: Tm) -> bool {
    match t {
        Tm::C { .. } => true,
        Tm::P { .. } => false,
    }
}

// ----------------------------------------------------------------------------
// Single-step Reduction (Simplified - Direct Definition)
// ----------------------------------------------------------------------------

// Direct step relation without existentials for simpler verification
pub open spec fn step_const_const(n1: int, n2: int, t2: Tm) -> bool {
    t2 == Tm::C { n: n1 + n2 }
}

// Check if a term is a constant addition that can reduce
pub open spec fn can_reduce_pcc(t: Tm) -> bool {
    match t {
        Tm::P { t1, t2 } => {
            match (*t1, *t2) {
                (Tm::C { .. }, Tm::C { .. }) => true,
                _ => false,
            }
        }
        _ => false,
    }
}

// Get result of reducing C n1 + C n2
pub open spec fn reduce_pcc(t: Tm) -> Tm {
    match t {
        Tm::P { t1, t2 } => {
            match (*t1, *t2) {
                (Tm::C { n: n1 }, Tm::C { n: n2 }) => Tm::C { n: n1 + n2 },
                _ => t,  // unreachable if can_reduce_pcc is true
            }
        }
        _ => t,
    }
}

// ----------------------------------------------------------------------------
// Multi-step Reduction (using fuel)
// ----------------------------------------------------------------------------

pub open spec fn eval(t: Tm, fuel: nat) -> Tm
    decreases fuel
{
    if fuel == 0 {
        t
    } else if can_reduce_pcc(t) {
        eval(reduce_pcc(t), (fuel - 1) as nat)
    } else {
        match t {
            Tm::P { t1, t2 } => {
                // Try to reduce left side first
                let t1_reduced = eval(*t1, (fuel - 1) as nat);
                if t1_reduced != *t1 {
                    Tm::P { t1: Box::new(t1_reduced), t2: t2 }
                } else {
                    // Then try right side
                    let t2_reduced = eval(*t2, (fuel - 1) as nat);
                    Tm::P { t1: t1, t2: Box::new(t2_reduced) }
                }
            }
            _ => t,
        }
    }
}

// ----------------------------------------------------------------------------
// Normal Forms and Values
// ----------------------------------------------------------------------------

// Values are normal forms
pub proof fn value_is_normal(t: Tm)
    requires value(t)
    ensures !can_reduce_pcc(t)
{
    match t {
        Tm::C { .. } => {}
        Tm::P { .. } => {}  // Not a value, so precondition is false
    }
}

// ----------------------------------------------------------------------------
// Size (for termination arguments)
// ----------------------------------------------------------------------------

pub open spec fn tm_size(t: Tm) -> nat
    decreases t
{
    match t {
        Tm::C { .. } => 1,
        Tm::P { t1, t2 } => 1 + tm_size(*t1) + tm_size(*t2),
    }
}

// Reduction of constants decreases size
pub proof fn reduce_pcc_decreases_size(t: Tm)
    requires can_reduce_pcc(t)
    ensures tm_size(reduce_pcc(t)) < tm_size(t)
{
    reveal_with_fuel(tm_size, 3);
    match t {
        Tm::P { t1, t2 } => {
            match (*t1, *t2) {
                (Tm::C { n: _ }, Tm::C { n: _ }) => {
                    // P(C n1, C n2) reduces to C(n1+n2)
                    // Size goes from 1 + 1 + 1 = 3 to 1
                }
                _ => {}
            }
        }
        _ => {}
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

// Example: 1 + 2 can reduce
pub proof fn example_can_reduce()
    ensures can_reduce_pcc(Tm::P { t1: Box::new(Tm::C { n: 1 }), t2: Box::new(Tm::C { n: 2 }) })
{
}

// Example: 1 + 2 reduces to 3
pub proof fn example_reduce_1_plus_2()
    ensures reduce_pcc(Tm::P { t1: Box::new(Tm::C { n: 1 }), t2: Box::new(Tm::C { n: 2 }) })
        == (Tm::C { n: 3 })
{
}

// Example: Constants are values
pub proof fn example_const_value()
{
    assert(value(Tm::C { n: 42 }));
    assert(!value(Tm::P { t1: Box::new(Tm::C { n: 1 }), t2: Box::new(Tm::C { n: 2 }) }));
}

// Example: Size calculation
pub proof fn example_size()
{
    reveal_with_fuel(tm_size, 3);
    assert(tm_size(Tm::C { n: 5 }) == 1);
    let t = Tm::P { t1: Box::new(Tm::C { n: 1 }), t2: Box::new(Tm::C { n: 2 }) };
    assert(tm_size(t) == 3);
}

// Example: Evaluation with fuel
pub proof fn example_eval()
{
    let t = Tm::P { t1: Box::new(Tm::C { n: 1 }), t2: Box::new(Tm::C { n: 2 }) };
    reveal_with_fuel(eval, 2);
    assert(eval(t, 1) == Tm::C { n: 3 });
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn smallstep_examples_verify()
{
    example_can_reduce();
    example_reduce_1_plus_2();
    example_const_value();
    example_size();
    example_eval();

    // Test reduce decreases size
    let t = Tm::P { t1: Box::new(Tm::C { n: 1 }), t2: Box::new(Tm::C { n: 2 }) };
    reduce_pcc_decreases_size(t);
}

pub fn main() {
    proof {
        smallstep_examples_verify();
    }
}

} // verus!
