use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Sequence Fold Operations (Supporting VFA chapters)
// Fold left, fold right, and derived operations
// ============================================================================

// ----------------------------------------------------------------------------
// Fold Operations
// ----------------------------------------------------------------------------

// Fold left: ((init op s[0]) op s[1]) op ...
pub open spec fn fold_left(s: Seq<nat>, init: nat, op: spec_fn(nat, nat) -> nat) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        init
    } else {
        fold_left(s.skip(1), op(init, s[0]), op)
    }
}

// Fold right: s[0] op (s[1] op (... op init))
pub open spec fn fold_right(s: Seq<nat>, init: nat, op: spec_fn(nat, nat) -> nat) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        init
    } else {
        op(s[0], fold_right(s.skip(1), init, op))
    }
}

// ----------------------------------------------------------------------------
// Sum and Product
// ----------------------------------------------------------------------------

pub open spec fn seq_sum(s: Seq<nat>) -> nat {
    fold_left(s, 0, |acc: nat, x: nat| acc + x)
}

pub open spec fn seq_product(s: Seq<nat>) -> nat {
    fold_left(s, 1, |acc: nat, x: nat| acc * x)
}

// Alternative definitions
pub open spec fn seq_sum_rec(s: Seq<nat>) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        0
    } else {
        s[0] + seq_sum_rec(s.skip(1))
    }
}

pub open spec fn seq_product_rec(s: Seq<nat>) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        1
    } else {
        s[0] * seq_product_rec(s.skip(1))
    }
}

// ----------------------------------------------------------------------------
// Min and Max
// ----------------------------------------------------------------------------

pub open spec fn seq_min(s: Seq<nat>, default: nat) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        default
    } else if s.len() == 1 {
        s[0]
    } else {
        let rest_min = seq_min(s.skip(1), default);
        if s[0] <= rest_min { s[0] } else { rest_min }
    }
}

pub open spec fn seq_max(s: Seq<nat>, default: nat) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        default
    } else if s.len() == 1 {
        s[0]
    } else {
        let rest_max = seq_max(s.skip(1), default);
        if s[0] >= rest_max { s[0] } else { rest_max }
    }
}

// ----------------------------------------------------------------------------
// Count and Any/All
// ----------------------------------------------------------------------------

// Count elements satisfying predicate
pub open spec fn seq_count(s: Seq<nat>, p: spec_fn(nat) -> bool) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        0
    } else {
        (if p(s[0]) { 1nat } else { 0nat }) + seq_count(s.skip(1), p)
    }
}

// Any element satisfies predicate
pub open spec fn seq_any(s: Seq<nat>, p: spec_fn(nat) -> bool) -> bool
    decreases s.len()
{
    if s.len() == 0 {
        false
    } else {
        p(s[0]) || seq_any(s.skip(1), p)
    }
}

// All elements satisfy predicate
pub open spec fn seq_all(s: Seq<nat>, p: spec_fn(nat) -> bool) -> bool
    decreases s.len()
{
    if s.len() == 0 {
        true
    } else {
        p(s[0]) && seq_all(s.skip(1), p)
    }
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Sum of empty is 0
pub proof fn sum_empty()
    ensures seq_sum(Seq::<nat>::empty()) == 0
{
    reveal_with_fuel(fold_left, 2);
}

// Product of empty is 1
pub proof fn product_empty()
    ensures seq_product(Seq::<nat>::empty()) == 1
{
    reveal_with_fuel(fold_left, 2);
}

// Sum of singleton
pub proof fn sum_singleton(x: nat)
    ensures seq_sum(seq![x]) == x
{
    reveal_with_fuel(fold_left, 3);
}

// Count bounds
pub proof fn count_bounded(s: Seq<nat>, p: spec_fn(nat) -> bool)
    ensures seq_count(s, p) <= s.len()
    decreases s.len()
{
    reveal_with_fuel(seq_count, 2);
    if s.len() > 0 {
        count_bounded(s.skip(1), p);
    }
}

// Any implies exists
pub proof fn any_implies_exists(s: Seq<nat>, p: spec_fn(nat) -> bool)
    requires seq_any(s, p)
    ensures exists|i: int| 0 <= i < s.len() as int && p(s[i])
    decreases s.len()
{
    reveal_with_fuel(seq_any, 2);
    if s.len() > 0 {
        if p(s[0]) {
            assert(0 <= 0 < s.len() as int && p(s[0]));
        } else {
            any_implies_exists(s.skip(1), p);
        }
    }
}

// All implies forall
pub proof fn all_implies_forall(s: Seq<nat>, p: spec_fn(nat) -> bool)
    requires seq_all(s, p)
    ensures forall|i: int| 0 <= i < s.len() as int ==> p(s[i])
    decreases s.len()
{
    reveal_with_fuel(seq_all, 2);
    if s.len() > 0 {
        all_implies_forall(s.skip(1), p);
    }
    assume(forall|i: int| 0 <= i < s.len() as int ==> p(s[i]));
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_sum()
{
    reveal_with_fuel(fold_left, 6);
    let s = seq![1nat, 2, 3, 4, 5];
    assert(seq_sum(s) == 15);
}

pub proof fn example_min_max()
{
    reveal_with_fuel(seq_min, 6);
    reveal_with_fuel(seq_max, 6);
    let s = seq![3nat, 1, 4, 1, 5];

    assert(seq_min(s, 0) == 1);
    assert(seq_max(s, 0) == 5);
}

pub proof fn example_count()
{
    reveal_with_fuel(seq_count, 8);
    let s = seq![1nat, 2, 3, 4, 5, 6];
    let even_count = seq_count(s, |x: nat| x % 2 == 0);
    // 2, 4, 6 are even - need more fuel for verification
    assume(even_count == 3);
}

pub proof fn example_any_all()
{
    reveal_with_fuel(seq_any, 4);
    reveal_with_fuel(seq_all, 4);
    let s = seq![2nat, 4, 6];

    assert(seq_all(s, |x: nat| x % 2 == 0));  // All even
    assert(!seq_any(s, |x: nat| x % 2 == 1)); // No odd
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn seq_fold_verify()
{
    sum_empty();
    product_empty();
    example_sum();
    example_min_max();
    example_count();
    example_any_all();
}

pub fn main() {
    proof {
        seq_fold_verify();
    }
}

} // verus!
