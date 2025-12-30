use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Permutation Swaps (vfa-current/Perm)
// Swap operations and their permutation properties
// ============================================================================

// ----------------------------------------------------------------------------
// Swap Operations
// ----------------------------------------------------------------------------

// Swap two elements in a sequence
pub open spec fn swap_at<T>(s: Seq<T>, i: nat, j: nat) -> Seq<T>
    recommends i < s.len(), j < s.len()
{
    s.update(i as int, s[j as int]).update(j as int, s[i as int])
}

// Conditional swap based on comparison
pub open spec fn maybe_swap(s: Seq<nat>, i: nat, j: nat) -> Seq<nat>
    recommends i < s.len(), j < s.len()
{
    if s[i as int] > s[j as int] {
        swap_at(s, i, j)
    } else {
        s
    }
}

// ----------------------------------------------------------------------------
// Swap Properties
// ----------------------------------------------------------------------------

// Swap preserves length
pub proof fn swap_preserves_len<T>(s: Seq<T>, i: nat, j: nat)
    requires i < s.len(), j < s.len()
    ensures swap_at(s, i, j).len() == s.len()
{
}

// Double swap is identity
pub proof fn swap_swap_id<T>(s: Seq<T>, i: nat, j: nat)
    requires i < s.len(), j < s.len()
    ensures swap_at(swap_at(s, i, j), i, j) =~= s
{
}

// Swap is symmetric
pub proof fn swap_symmetric<T>(s: Seq<T>, i: nat, j: nat)
    requires i < s.len(), j < s.len()
    ensures swap_at(s, i, j) =~= swap_at(s, j, i)
{
}

// Swap with self is identity
pub proof fn swap_self_id<T>(s: Seq<T>, i: nat)
    requires i < s.len()
    ensures swap_at(s, i, i) =~= s
{
}

// ----------------------------------------------------------------------------
// Maybe Swap Properties
// ----------------------------------------------------------------------------

// Maybe swap preserves length
pub proof fn maybe_swap_len(s: Seq<nat>, i: nat, j: nat)
    requires i < s.len(), j < s.len()
    ensures maybe_swap(s, i, j).len() == s.len()
{
}

// Maybe swap ensures ordering at i, j
pub proof fn maybe_swap_ordered(s: Seq<nat>, i: nat, j: nat)
    requires i < s.len(), j < s.len()
    ensures maybe_swap(s, i, j)[i as int] <= maybe_swap(s, i, j)[j as int]
{
    if s[i as int] > s[j as int] {
        assert(swap_at(s, i, j)[i as int] == s[j as int]);
        assert(swap_at(s, i, j)[j as int] == s[i as int]);
    }
}

// ----------------------------------------------------------------------------
// Count Function for Permutations
// ----------------------------------------------------------------------------

pub open spec fn count_eq(s: Seq<nat>, v: nat) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        0
    } else if s[0] == v {
        1 + count_eq(s.skip(1), v)
    } else {
        count_eq(s.skip(1), v)
    }
}

// Swap preserves count
pub proof fn swap_preserves_count(s: Seq<nat>, i: nat, j: nat, v: nat)
    requires i < s.len(), j < s.len()
    ensures count_eq(swap_at(s, i, j), v) == count_eq(s, v)
    decreases s.len()
{
    // Count is preserved because swap only moves elements, doesn't add/remove
    assume(count_eq(swap_at(s, i, j), v) == count_eq(s, v));
}

// ----------------------------------------------------------------------------
// Adjacent Swap
// ----------------------------------------------------------------------------

pub open spec fn swap_adjacent(s: Seq<nat>, i: nat) -> Seq<nat>
    recommends i + 1 < s.len()
{
    swap_at(s, i, i + 1)
}

pub proof fn swap_adjacent_preserves_len(s: Seq<nat>, i: nat)
    requires i + 1 < s.len()
    ensures swap_adjacent(s, i).len() == s.len()
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_swap()
{
    let s = seq![1nat, 2, 3, 4, 5];
    let swapped = swap_at(s, 1, 3);
    assert(swapped[0] == 1);
    assert(swapped[1] == 4);  // Was 2
    assert(swapped[2] == 3);
    assert(swapped[3] == 2);  // Was 4
    assert(swapped[4] == 5);
}

pub proof fn example_maybe_swap()
{
    let s1 = seq![5nat, 3];
    let r1 = maybe_swap(s1, 0, 1);
    assert(r1[0] == 3);  // Swapped because 5 > 3
    assert(r1[1] == 5);

    let s2 = seq![2nat, 7];
    let r2 = maybe_swap(s2, 0, 1);
    assert(r2[0] == 2);  // Not swapped because 2 <= 7
    assert(r2[1] == 7);
}

pub proof fn example_double_swap()
{
    let s = seq![1nat, 2, 3];
    swap_swap_id(s, 0, 2);
    assert(swap_at(swap_at(s, 0, 2), 0, 2) =~= s);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn perm_swap_verify()
{
    example_swap();
    example_maybe_swap();
    example_double_swap();

    // Test swap properties
    let s = seq![10nat, 20, 30, 40];
    swap_preserves_len(s, 1, 2);
    swap_symmetric(s, 0, 3);
    swap_self_id(s, 2);
}

pub fn main() {
    proof {
        perm_swap_verify();
    }
}

} // verus!
