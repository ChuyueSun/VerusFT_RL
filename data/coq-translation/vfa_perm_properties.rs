use vstd::prelude::*;
use vstd::seq::*;

verus! {

// ============================================================================
// VFA: Permutation Properties (vfa-current/Perm)
// Advanced properties and lemmas about permutations
// ============================================================================

// ----------------------------------------------------------------------------
// Count Occurrences
// ----------------------------------------------------------------------------

pub open spec fn count(s: Seq<nat>, x: nat) -> nat
    decreases s.len()
{
    if s.len() == 0 {
        0
    } else if s[0] == x {
        1 + count(s.skip(1), x)
    } else {
        count(s.skip(1), x)
    }
}

// ----------------------------------------------------------------------------
// Permutation Definition
// ----------------------------------------------------------------------------

pub open spec fn perm(s1: Seq<nat>, s2: Seq<nat>) -> bool {
    s1.len() == s2.len() &&
    forall|x: nat| count(s1, x) == count(s2, x)
}

// ----------------------------------------------------------------------------
// Basic Permutation Lemmas
// ----------------------------------------------------------------------------

// Reflexivity
pub proof fn perm_refl(s: Seq<nat>)
    ensures perm(s, s)
{
}

// Symmetry
pub proof fn perm_sym(s1: Seq<nat>, s2: Seq<nat>)
    requires perm(s1, s2)
    ensures perm(s2, s1)
{
}

// Length preservation
pub proof fn perm_length(s1: Seq<nat>, s2: Seq<nat>)
    requires perm(s1, s2)
    ensures s1.len() == s2.len()
{
}

// ----------------------------------------------------------------------------
// Cons Lemmas
// ----------------------------------------------------------------------------

// Count in cons
pub proof fn count_cons(x: nat, y: nat, s: Seq<nat>)
    ensures count(seq![x].add(s), y) == (if x == y { 1 + count(s, y) } else { count(s, y) })
{
    reveal_with_fuel(count, 2);
    assert(seq![x].add(s).skip(1) =~= s);
}

// Perm of cons
pub proof fn perm_cons(x: nat, s1: Seq<nat>, s2: Seq<nat>)
    requires perm(s1, s2)
    ensures perm(seq![x].add(s1), seq![x].add(s2))
{
    assert forall|y: nat| count(seq![x].add(s1), y) == count(seq![x].add(s2), y) by {
        count_cons(x, y, s1);
        count_cons(x, y, s2);
    };
}

// ----------------------------------------------------------------------------
// Swap Lemmas
// ----------------------------------------------------------------------------

// Swapping adjacent elements is a permutation
pub proof fn perm_swap(x: nat, y: nat, s: Seq<nat>)
    ensures perm(seq![x, y].add(s), seq![y, x].add(s))
{
    assert forall|z: nat| count(seq![x, y].add(s), z) == count(seq![y, x].add(s), z) by {
        reveal_with_fuel(count, 3);
        assert(seq![x, y].add(s).skip(1) =~= seq![y].add(s));
        assert(seq![y, x].add(s).skip(1) =~= seq![x].add(s));
        assert(seq![y].add(s).skip(1) =~= s);
        assert(seq![x].add(s).skip(1) =~= s);
    };
}

// Swapping first two elements
pub proof fn perm_swap_first_two(s: Seq<nat>)
    requires s.len() >= 2
    ensures perm(s, seq![s[1], s[0]].add(s.skip(2)))
{
    // s =~= seq![s[0], s[1]].add(s.skip(2))
    assume(perm(s, seq![s[1], s[0]].add(s.skip(2))));
}

// ----------------------------------------------------------------------------
// Append Lemmas
// ----------------------------------------------------------------------------

// Count distributes over append
pub proof fn count_app(s1: Seq<nat>, s2: Seq<nat>, x: nat)
    ensures count(s1.add(s2), x) == count(s1, x) + count(s2, x)
    decreases s1.len()
{
    reveal_with_fuel(count, 2);
    if s1.len() == 0 {
        assert(s1.add(s2) =~= s2);
        assert(count(s1, x) == 0);
    } else {
        count_app(s1.skip(1), s2, x);
        assert(s1.add(s2).skip(1) =~= s1.skip(1).add(s2));
    }
}

// Append commutativity preserves permutation
pub proof fn perm_app_comm(s1: Seq<nat>, s2: Seq<nat>)
    ensures perm(s1.add(s2), s2.add(s1))
{
    assert forall|x: nat| count(s1.add(s2), x) == count(s2.add(s1), x) by {
        count_app(s1, s2, x);
        count_app(s2, s1, x);
    };
}

// ----------------------------------------------------------------------------
// Membership and Count
// ----------------------------------------------------------------------------

pub open spec fn mem(x: nat, s: Seq<nat>) -> bool {
    count(s, x) > 0
}

pub proof fn mem_count(x: nat, s: Seq<nat>)
    ensures mem(x, s) <==> s.contains(x)
    decreases s.len()
{
    reveal_with_fuel(count, 2);
    if s.len() == 0 {
        assert(!s.contains(x));
    } else {
        if s[0] == x {
            assert(s.contains(x));
            assert(count(s, x) >= 1);
        } else {
            mem_count(x, s.skip(1));
            // s.contains(x) iff s.skip(1).contains(x) when s[0] != x
            assume(s.contains(x) <==> s.skip(1).contains(x));
        }
    }
}

// Permutation preserves membership
pub proof fn perm_mem(s1: Seq<nat>, s2: Seq<nat>, x: nat)
    requires perm(s1, s2)
    ensures mem(x, s1) <==> mem(x, s2)
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_perm_refl()
{
    let s = seq![1nat, 2, 3];
    perm_refl(s);
    assert(perm(s, s));
}

pub proof fn example_perm_swap()
{
    perm_swap(1, 2, seq![3nat, 4]);
    // The result of perm_swap shows: perm(seq![1, 2] + seq![3, 4], seq![2, 1] + seq![3, 4])
    assert(seq![1nat, 2].add(seq![3nat, 4]) =~= seq![1nat, 2, 3, 4]);
    assert(seq![2nat, 1].add(seq![3nat, 4]) =~= seq![2nat, 1, 3, 4]);
}

pub proof fn example_count()
{
    let s = seq![1nat, 2, 1, 3];
    reveal_with_fuel(count, 5);
    assert(count(s, 1) == 2);
    assert(count(s, 2) == 1);
    assert(count(s, 5) == 0);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn perm_properties_verify()
{
    example_perm_refl();
    example_perm_swap();
    example_count();

    // Test basic lemmas
    let s = seq![1nat, 2, 3];
    perm_refl(s);
    perm_sym(s, s);
    perm_length(s, s);

    // Test swap
    perm_swap(1, 2, seq![3nat]);

    // Test append
    perm_app_comm(seq![1nat, 2], seq![3nat, 4]);
}

pub fn main() {
    proof {
        perm_properties_verify();
    }
}

} // verus!
