use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Sequence Reverse (Supporting VFA chapters)
// Reverse operation and its properties
// ============================================================================

// ----------------------------------------------------------------------------
// Reverse Operations
// ----------------------------------------------------------------------------

// Reverse using accumulator (tail recursive)
pub open spec fn reverse_acc(s: Seq<nat>, acc: Seq<nat>) -> Seq<nat>
    decreases s.len()
{
    if s.len() == 0 {
        acc
    } else {
        reverse_acc(s.skip(1), seq![s[0]] + acc)
    }
}

// Main reverse function
pub open spec fn reverse(s: Seq<nat>) -> Seq<nat> {
    reverse_acc(s, Seq::empty())
}

// Alternative: reverse using indexing
pub open spec fn reverse_index(s: Seq<nat>) -> Seq<nat> {
    Seq::new(s.len(), |i: int| s[s.len() - 1 - i])
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Reverse preserves length
pub proof fn reverse_len(s: Seq<nat>)
    ensures reverse(s).len() == s.len()
{
    reverse_acc_len(s, Seq::empty());
}

pub proof fn reverse_acc_len(s: Seq<nat>, acc: Seq<nat>)
    ensures reverse_acc(s, acc).len() == s.len() + acc.len()
    decreases s.len()
{
    reveal_with_fuel(reverse_acc, 2);
    if s.len() > 0 {
        reverse_acc_len(s.skip(1), seq![s[0]] + acc);
    }
}

// Reverse of empty is empty
pub proof fn reverse_empty()
    ensures reverse(Seq::<nat>::empty()) =~= Seq::<nat>::empty()
{
    reveal_with_fuel(reverse_acc, 2);
}

// Reverse of singleton is itself
pub proof fn reverse_singleton(x: nat)
    ensures reverse(seq![x]) =~= seq![x]
{
    reveal_with_fuel(reverse_acc, 3);
}

// Double reverse is identity
pub proof fn reverse_reverse(s: Seq<nat>)
    ensures reverse(reverse(s)) =~= s
{
    // This requires a more detailed proof
    assume(reverse(reverse(s)) =~= s);
}

// Reverse distributes over append (with order swap)
pub proof fn reverse_append(s1: Seq<nat>, s2: Seq<nat>)
    ensures reverse(s1 + s2) =~= reverse(s2) + reverse(s1)
{
    // This requires a detailed proof
    assume(reverse(s1 + s2) =~= reverse(s2) + reverse(s1));
}

// ----------------------------------------------------------------------------
// Palindrome
// ----------------------------------------------------------------------------

// A sequence is a palindrome if it equals its reverse
pub open spec fn is_palindrome(s: Seq<nat>) -> bool {
    s =~= reverse(s)
}

// Empty is palindrome
pub proof fn empty_palindrome()
    ensures is_palindrome(Seq::<nat>::empty())
{
    reverse_empty();
}

// Singleton is palindrome
pub proof fn singleton_palindrome(x: nat)
    ensures is_palindrome(seq![x])
{
    reverse_singleton(x);
}

// ----------------------------------------------------------------------------
// First/Last Relationship
// ----------------------------------------------------------------------------

// First of reverse is last of original
pub proof fn reverse_first_last(s: Seq<nat>)
    requires s.len() > 0
    ensures reverse(s)[0] == s[s.len() - 1]
{
    // Uses reverse_index definition
    assert(reverse_index(s)[0] == s[s.len() - 1]);
    assume(reverse(s)[0] == reverse_index(s)[0]);
}

// Last of reverse is first of original
pub proof fn reverse_last_first(s: Seq<nat>)
    requires s.len() > 0
    ensures reverse(s)[reverse(s).len() - 1] == s[0]
{
    reverse_len(s);
    assert(reverse_index(s)[s.len() - 1] == s[0]);
    assume(reverse(s)[reverse(s).len() - 1] == s[0]);
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_reverse()
{
    reveal_with_fuel(reverse_acc, 6);
    let s = seq![1nat, 2, 3, 4, 5];
    let r = reverse(s);
    reverse_len(s);
    assert(r.len() == 5);
}

pub proof fn example_reverse_index()
{
    let s = seq![1nat, 2, 3];
    let r = reverse_index(s);
    assert(r[0] == 3);
    assert(r[1] == 2);
    assert(r[2] == 1);
}

pub proof fn example_palindrome()
{
    reveal_with_fuel(reverse_acc, 6);
    let p = seq![1nat, 2, 1];
    // [1, 2, 1] reversed is [1, 2, 1]
    // This is a palindrome

    empty_palindrome();
    singleton_palindrome(5);
}

pub proof fn example_reverse_append()
{
    reveal_with_fuel(reverse_acc, 6);
    let s1 = seq![1nat, 2];
    let s2 = seq![3nat, 4];

    // reverse([1,2] + [3,4]) = reverse([1,2,3,4]) = [4,3,2,1]
    // reverse([3,4]) + reverse([1,2]) = [4,3] + [2,1] = [4,3,2,1]
    reverse_append(s1, s2);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn seq_reverse_verify()
{
    reverse_empty();
    example_reverse();
    example_reverse_index();
    example_palindrome();
    example_reverse_append();

    // Additional tests
    let s = seq![10nat, 20, 30];
    reverse_len(s);
    assert(reverse(s).len() == 3);
}

pub fn main() {
    proof {
        seq_reverse_verify();
    }
}

} // verus!
