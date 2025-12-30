use vstd::prelude::*;

verus! {

// Extraction (Software Foundations, LF/Extraction) style snippets in Verus.
//
// SF's theme: you write a specification in a proof-oriented language, then "extract"
// an executable program and argue it matches the spec.
//
// Verus' analogue: write a `spec fn` (mathematical model) and an `exec fn` (Rust code)
// with `ensures` tying the executable result back to the spec.

// ----------------------------
// A spec function (math model)
// ----------------------------

pub open spec fn triangle(n: nat) -> nat
    decreases n,
{
    if n == 0 {
        0
    } else {
        n + triangle((n - 1) as nat)
    }
}

// A simple proof used to justify overflow checks in the executable code.
pub proof fn triangle_is_monotonic(i: nat, j: nat)
    ensures i <= j ==> triangle(i) <= triangle(j)
    decreases j,
{
    if j == 0 {
    } else {
        triangle_is_monotonic(i, (j - 1) as nat);
    }
}

// ----------------------------
// Executable code + correctness link
// ----------------------------

pub fn loop_triangle(n: u32) -> (sum: u32)
    requires triangle(n as nat) < 0x1_0000_0000,
    ensures sum == triangle(n as nat),
{
    let mut sum: u32 = 0;
    let mut idx: u32 = 0;

    while idx < n
        invariant
            idx <= n,
            sum == triangle(idx as nat),
            triangle(n as nat) < 0x1_0000_0000,
        decreases n - idx,
    {
        idx = idx + 1;
        assert(sum + idx < 0x1_0000_0000) by {
            triangle_is_monotonic(idx as nat, n as nat);
        }
        sum = sum + idx;
    }

    sum
}

pub fn main() {
    // Establish the precondition for a tiny input, then run the extracted program.
    assert(triangle(1) < 0x1_0000_0000) by {
        reveal_with_fuel(triangle, 2);
    }
    let _s = loop_triangle(1);
}

} // verus!
