use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Sequence Basics (Supporting VFA chapters)
// Basic sequence operations and properties
// ============================================================================

// ----------------------------------------------------------------------------
// Sequence Constructors
// ----------------------------------------------------------------------------

// Create sequence of n copies of x
pub open spec fn seq_repeat<T>(x: T, n: nat) -> Seq<T> {
    Seq::new(n, |i: int| x)
}

// Create sequence [0, 1, 2, ..., n-1]
pub open spec fn seq_range(n: nat) -> Seq<nat> {
    Seq::new(n, |i: int| i as nat)
}

// Create sequence [start, start+1, ..., end-1]
pub open spec fn seq_range_from(start: nat, end: nat) -> Seq<nat>
    recommends start <= end
{
    Seq::new((end - start) as nat, |i: int| (start + i) as nat)
}

// ----------------------------------------------------------------------------
// Basic Operations
// ----------------------------------------------------------------------------

// Sequence length
pub open spec fn seq_len<T>(s: Seq<T>) -> nat {
    s.len()
}

// Get element at index
pub open spec fn seq_get<T>(s: Seq<T>, i: nat) -> T
    recommends i < s.len()
{
    s[i as int]
}

// Update element at index
pub open spec fn seq_set<T>(s: Seq<T>, i: nat, x: T) -> Seq<T>
    recommends i < s.len()
{
    s.update(i as int, x)
}

// First element
pub open spec fn seq_first<T>(s: Seq<T>) -> T
    recommends s.len() > 0
{
    s[0]
}

// Last element
pub open spec fn seq_last<T>(s: Seq<T>) -> T
    recommends s.len() > 0
{
    s[s.len() - 1]
}

// ----------------------------------------------------------------------------
// Subsequence Operations
// ----------------------------------------------------------------------------

// Take first n elements
pub open spec fn seq_take<T>(s: Seq<T>, n: nat) -> Seq<T> {
    s.take(n as int)
}

// Drop first n elements
pub open spec fn seq_drop<T>(s: Seq<T>, n: nat) -> Seq<T> {
    s.skip(n as int)
}

// Slice from i to j
pub open spec fn seq_slice<T>(s: Seq<T>, i: nat, j: nat) -> Seq<T>
    recommends i <= j && j <= s.len()
{
    s.subrange(i as int, j as int)
}

// ----------------------------------------------------------------------------
// Concatenation
// ----------------------------------------------------------------------------

pub open spec fn seq_append<T>(s1: Seq<T>, s2: Seq<T>) -> Seq<T> {
    s1 + s2
}

pub open spec fn seq_cons<T>(x: T, s: Seq<T>) -> Seq<T> {
    seq![x] + s
}

pub open spec fn seq_snoc<T>(s: Seq<T>, x: T) -> Seq<T> {
    s.push(x)
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Repeat creates sequence of correct length
pub proof fn repeat_len<T>(x: T, n: nat)
    ensures seq_repeat(x, n).len() == n
{
}

// Range creates sequence of correct length
pub proof fn range_len(n: nat)
    ensures seq_range(n).len() == n
{
}

// Range elements are correct
pub proof fn range_elements(n: nat, i: nat)
    requires i < n
    ensures seq_range(n)[i as int] == i
{
}

// Append length
pub proof fn append_len<T>(s1: Seq<T>, s2: Seq<T>)
    ensures seq_append(s1, s2).len() == s1.len() + s2.len()
{
}

// Take and drop split sequence
pub proof fn take_drop_split<T>(s: Seq<T>, n: nat)
    requires n <= s.len()
    ensures seq_append(seq_take(s, n), seq_drop(s, n)) =~= s
{
}

// Cons increases length by 1
pub proof fn cons_len<T>(x: T, s: Seq<T>)
    ensures seq_cons(x, s).len() == s.len() + 1
{
}

// Snoc increases length by 1
pub proof fn snoc_len<T>(s: Seq<T>, x: T)
    ensures seq_snoc(s, x).len() == s.len() + 1
{
}

// First of cons
pub proof fn cons_first<T>(x: T, s: Seq<T>)
    ensures seq_first(seq_cons(x, s)) == x
{
}

// Last of snoc
pub proof fn snoc_last<T>(s: Seq<T>, x: T)
    ensures seq_last(seq_snoc(s, x)) == x
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_repeat()
{
    let s = seq_repeat(5nat, 3);
    repeat_len(5nat, 3);
    assert(s.len() == 3);
    assert(s[0] == 5);
    assert(s[1] == 5);
    assert(s[2] == 5);
}

pub proof fn example_range()
{
    let s = seq_range(5);
    range_len(5);
    assert(s.len() == 5);
    range_elements(5, 0);
    range_elements(5, 2);
    range_elements(5, 4);
    assert(s[0] == 0);
    assert(s[2] == 2);
    assert(s[4] == 4);
}

pub proof fn example_take_drop()
{
    let s = seq![1nat, 2, 3, 4, 5];
    let t = seq_take(s, 3);
    let d = seq_drop(s, 3);

    assert(t.len() == 3);
    assert(d.len() == 2);

    take_drop_split(s, 3);
    assert(seq_append(t, d) =~= s);
}

pub proof fn example_cons_snoc()
{
    let s = seq![2nat, 3, 4];

    let c = seq_cons(1nat, s);
    cons_len(1nat, s);
    cons_first(1nat, s);
    assert(c.len() == 4);
    assert(seq_first(c) == 1);

    let sn = seq_snoc(s, 5nat);
    snoc_len(s, 5nat);
    snoc_last(s, 5nat);
    assert(sn.len() == 4);
    assert(seq_last(sn) == 5);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn seq_basics_verify()
{
    example_repeat();
    example_range();
    example_take_drop();
    example_cons_snoc();

    // Additional tests
    append_len(seq![1nat, 2], seq![3nat, 4, 5]);
}

pub fn main() {
    proof {
        seq_basics_verify();
    }
}

} // verus!
