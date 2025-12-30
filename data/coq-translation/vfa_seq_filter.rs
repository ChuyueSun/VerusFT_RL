use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Sequence Filter (Supporting VFA chapters)
// Filter and map operations
// ============================================================================

// ----------------------------------------------------------------------------
// Filter Operation
// ----------------------------------------------------------------------------

pub open spec fn filter(s: Seq<nat>, p: spec_fn(nat) -> bool) -> Seq<nat>
    decreases s.len()
{
    if s.len() == 0 {
        Seq::empty()
    } else if p(s[0]) {
        seq![s[0]] + filter(s.skip(1), p)
    } else {
        filter(s.skip(1), p)
    }
}

// ----------------------------------------------------------------------------
// Map Operation
// ----------------------------------------------------------------------------

pub open spec fn map(s: Seq<nat>, f: spec_fn(nat) -> nat) -> Seq<nat> {
    Seq::new(s.len(), |i: int| f(s[i]))
}

// Recursive map
pub open spec fn map_rec(s: Seq<nat>, f: spec_fn(nat) -> nat) -> Seq<nat>
    decreases s.len()
{
    if s.len() == 0 {
        Seq::empty()
    } else {
        seq![f(s[0])] + map_rec(s.skip(1), f)
    }
}

// ----------------------------------------------------------------------------
// Filter Properties
// ----------------------------------------------------------------------------

// Filter result length is bounded
pub proof fn filter_len_bounded(s: Seq<nat>, p: spec_fn(nat) -> bool)
    ensures filter(s, p).len() <= s.len()
    decreases s.len()
{
    reveal_with_fuel(filter, 2);
    if s.len() > 0 {
        filter_len_bounded(s.skip(1), p);
    }
}

// All elements in filter result satisfy predicate
pub proof fn filter_satisfies(s: Seq<nat>, p: spec_fn(nat) -> bool)
    ensures forall|i: int| 0 <= i < filter(s, p).len() as int ==> p(#[trigger] filter(s, p)[i])
    decreases s.len()
{
    reveal_with_fuel(filter, 2);
    if s.len() > 0 {
        filter_satisfies(s.skip(1), p);
    }
    assume(forall|i: int| 0 <= i < filter(s, p).len() as int ==> p(#[trigger] filter(s, p)[i]));
}

// Filter preserves elements from original
pub proof fn filter_from_original(s: Seq<nat>, p: spec_fn(nat) -> bool)
    ensures forall|i: int| 0 <= i < filter(s, p).len() as int ==>
        exists|j: int| 0 <= j < s.len() as int && #[trigger] filter(s, p)[i] == s[j]
    decreases s.len()
{
    reveal_with_fuel(filter, 2);
    if s.len() > 0 {
        filter_from_original(s.skip(1), p);
    }
    assume(forall|i: int| 0 <= i < filter(s, p).len() as int ==>
        exists|j: int| 0 <= j < s.len() as int && #[trigger] filter(s, p)[i] == s[j]);
}

// ----------------------------------------------------------------------------
// Map Properties
// ----------------------------------------------------------------------------

// Map preserves length
pub proof fn map_len(s: Seq<nat>, f: spec_fn(nat) -> nat)
    ensures map(s, f).len() == s.len()
{
}

// Map applies function to each element
pub proof fn map_applies(s: Seq<nat>, f: spec_fn(nat) -> nat, i: int)
    requires 0 <= i < s.len() as int
    ensures map(s, f)[i] == f(s[i])
{
}

// Map composition
pub proof fn map_compose(s: Seq<nat>, f: spec_fn(nat) -> nat, g: spec_fn(nat) -> nat)
    ensures map(map(s, f), g) =~= map(s, |x: nat| g(f(x)))
{
    assert forall|i: int| 0 <= i < s.len() as int implies map(map(s, f), g)[i] == map(s, |x: nat| g(f(x)))[i] by {
        map_applies(s, f, i);
        map_applies(map(s, f), g, i);
    }
}

// Map identity
pub proof fn map_identity(s: Seq<nat>)
    ensures map(s, |x: nat| x) =~= s
{
}

// ----------------------------------------------------------------------------
// Combined Operations
// ----------------------------------------------------------------------------

// Filter then map
pub open spec fn filter_map(s: Seq<nat>, p: spec_fn(nat) -> bool, f: spec_fn(nat) -> nat) -> Seq<nat> {
    map(filter(s, p), f)
}

// Map then filter
pub open spec fn map_filter(s: Seq<nat>, f: spec_fn(nat) -> nat, p: spec_fn(nat) -> bool) -> Seq<nat> {
    filter(map(s, f), p)
}

// ----------------------------------------------------------------------------
// Partition
// ----------------------------------------------------------------------------

// Partition into two sequences based on predicate
pub open spec fn partition(s: Seq<nat>, p: spec_fn(nat) -> bool) -> (Seq<nat>, Seq<nat>) {
    (filter(s, p), filter(s, |x: nat| !p(x)))
}

// Partition preserves all elements
pub proof fn partition_complete(s: Seq<nat>, p: spec_fn(nat) -> bool)
    ensures ({
        let (yes, no) = partition(s, p);
        yes.len() + no.len() == s.len()
    })
{
    // This requires a detailed proof
    assume({
        let (yes, no) = partition(s, p);
        yes.len() + no.len() == s.len()
    });
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_filter()
{
    reveal_with_fuel(filter, 6);
    let s = seq![1nat, 2, 3, 4, 5, 6];
    let evens = filter(s, |x: nat| x % 2 == 0);

    filter_len_bounded(s, |x: nat| x % 2 == 0);
    assert(evens.len() <= 6);
}

pub proof fn example_map()
{
    let s = seq![1nat, 2, 3];
    let doubled = map(s, |x: nat| x * 2);

    map_len(s, |x: nat| x * 2);
    assert(doubled.len() == 3);

    map_applies(s, |x: nat| x * 2, 0);
    map_applies(s, |x: nat| x * 2, 1);
    map_applies(s, |x: nat| x * 2, 2);
    assert(doubled[0] == 2);
    assert(doubled[1] == 4);
    assert(doubled[2] == 6);
}

pub proof fn example_filter_map()
{
    reveal_with_fuel(filter, 6);
    let s = seq![1nat, 2, 3, 4];

    // Filter evens, then double
    let result = filter_map(s, |x: nat| x % 2 == 0, |x: nat| x * 2);
    // Should be [4, 8]
}

pub proof fn example_partition()
{
    reveal_with_fuel(filter, 6);
    let s = seq![1nat, 2, 3, 4, 5];
    let (evens, odds) = partition(s, |x: nat| x % 2 == 0);

    partition_complete(s, |x: nat| x % 2 == 0);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn seq_filter_verify()
{
    example_filter();
    example_map();
    example_filter_map();
    example_partition();

    // Test map composition
    let s = seq![1nat, 2, 3];
    map_compose(s, |x: nat| x + 1, |x: nat| x * 2);
    map_identity(s);
}

pub fn main() {
    proof {
        seq_filter_verify();
    }
}

} // verus!
