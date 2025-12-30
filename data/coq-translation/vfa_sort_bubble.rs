use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Bubble Sort (vfa-current/Sort)
// Bubble sort implementation and verification
// ============================================================================

// ----------------------------------------------------------------------------
// Sorted Predicate
// ----------------------------------------------------------------------------

pub open spec fn sorted(s: Seq<nat>) -> bool
    decreases s.len()
{
    if s.len() <= 1 {
        true
    } else {
        s[0] <= s[1] && sorted(s.skip(1))
    }
}

// ----------------------------------------------------------------------------
// Swap Operation
// ----------------------------------------------------------------------------

pub open spec fn swap_at(s: Seq<nat>, i: nat, j: nat) -> Seq<nat>
    recommends i < s.len(), j < s.len()
{
    s.update(i as int, s[j as int]).update(j as int, s[i as int])
}

// ----------------------------------------------------------------------------
// Bubble Pass - One pass through the list
// ----------------------------------------------------------------------------

// Single bubble pass: bubble largest to end
pub open spec fn bubble_pass(s: Seq<nat>, i: nat) -> Seq<nat>
    recommends i < s.len()
    decreases s.len() - i
{
    if i + 1 >= s.len() {
        s
    } else if s[i as int] > s[(i + 1) as int] {
        bubble_pass(swap_at(s, i, i + 1), i + 1)
    } else {
        bubble_pass(s, i + 1)
    }
}

// One complete pass from start
pub open spec fn one_pass(s: Seq<nat>) -> Seq<nat> {
    if s.len() <= 1 {
        s
    } else {
        bubble_pass(s, 0)
    }
}

// ----------------------------------------------------------------------------
// Bubble Sort
// ----------------------------------------------------------------------------

// Full bubble sort
pub open spec fn bubble_sort(s: Seq<nat>, passes: nat) -> Seq<nat>
    decreases passes
{
    if passes == 0 || s.len() <= 1 {
        s
    } else {
        bubble_sort(one_pass(s), (passes - 1) as nat)
    }
}

// Main sort function
pub open spec fn sort(s: Seq<nat>) -> Seq<nat> {
    bubble_sort(s, s.len())
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Bubble pass preserves length
pub proof fn bubble_pass_len(s: Seq<nat>, i: nat)
    requires i < s.len()
    ensures bubble_pass(s, i).len() == s.len()
    decreases s.len() - i
{
    reveal_with_fuel(bubble_pass, 3);
    if i + 1 < s.len() {
        if s[i as int] > s[(i + 1) as int] {
            bubble_pass_len(swap_at(s, i, i + 1), i + 1);
        } else {
            bubble_pass_len(s, i + 1);
        }
    }
}

// One pass preserves length
pub proof fn one_pass_len(s: Seq<nat>)
    ensures one_pass(s).len() == s.len()
{
    if s.len() > 1 {
        bubble_pass_len(s, 0);
    }
}

// Bubble sort preserves length
pub proof fn bubble_sort_len(s: Seq<nat>, passes: nat)
    ensures bubble_sort(s, passes).len() == s.len()
    decreases passes
{
    reveal_with_fuel(bubble_sort, 2);
    if passes > 0 && s.len() > 1 {
        one_pass_len(s);
        bubble_sort_len(one_pass(s), (passes - 1) as nat);
    }
}

// Sort preserves length
pub proof fn sort_len(s: Seq<nat>)
    ensures sort(s).len() == s.len()
{
    bubble_sort_len(s, s.len());
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_sorted()
{
    reveal_with_fuel(sorted, 4);
    assert(sorted(Seq::<nat>::empty()));
    assert(sorted(seq![5nat]));
    assert(sorted(seq![1nat, 2, 3]));
    assert(!sorted(seq![3nat, 1, 2]));
}

pub proof fn example_swap()
{
    let s = seq![5nat, 3, 4];
    let t = swap_at(s, 0, 1);
    assert(t[0] == 3);
    assert(t[1] == 5);
    assert(t[2] == 4);
}

pub proof fn example_bubble_simple()
{
    reveal_with_fuel(bubble_pass, 3);
    let s = seq![3nat, 1];
    let t = bubble_pass(s, 0);
    assert(t[0] == 1);
    assert(t[1] == 3);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn bubble_sort_verify()
{
    example_sorted();
    example_swap();
    example_bubble_simple();

    // Test length preservation
    let s = seq![5nat, 3, 8, 1];
    sort_len(s);
    assert(sort(s).len() == 4);
}

pub fn main() {
    proof {
        bubble_sort_verify();
    }
}

} // verus!
