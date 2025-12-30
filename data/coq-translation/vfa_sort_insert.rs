use vstd::prelude::*;
use vstd::seq::*;

verus! {

// ============================================================================
// VFA: Insertion Sort (vfa-current/Sort)
// Implementation and verification of insertion sort
// ============================================================================

// ----------------------------------------------------------------------------
// Sorted Predicate
// ----------------------------------------------------------------------------

// A sequence is sorted if every adjacent pair is in order
pub open spec fn sorted(s: Seq<nat>) -> bool
    decreases s.len()
{
    if s.len() <= 1 {
        true
    } else {
        s[0] <= s[1] && sorted(s.skip(1))
    }
}

// Alternative: all elements at index i are <= elements at index j where i < j
pub open spec fn sorted_alt(s: Seq<nat>) -> bool {
    forall|i: int, j: int| 0 <= i < j < s.len() ==> s[i] <= s[j]
}

// ----------------------------------------------------------------------------
// Insert Operation
// ----------------------------------------------------------------------------

// Insert an element into a sorted sequence maintaining sorted order
pub open spec fn insert(x: nat, s: Seq<nat>) -> Seq<nat>
    decreases s.len()
{
    if s.len() == 0 {
        seq![x]
    } else if x <= s[0] {
        seq![x].add(s)
    } else {
        seq![s[0]].add(insert(x, s.skip(1)))
    }
}

// ----------------------------------------------------------------------------
// Insertion Sort
// ----------------------------------------------------------------------------

// Sort a sequence using insertion sort
pub open spec fn insertion_sort(s: Seq<nat>) -> Seq<nat>
    decreases s.len()
{
    if s.len() == 0 {
        Seq::<nat>::empty()
    } else {
        insert(s[0], insertion_sort(s.skip(1)))
    }
}

// ----------------------------------------------------------------------------
// Correctness Properties
// ----------------------------------------------------------------------------

// Helper: sorted sequence with smaller element prepended is sorted
pub proof fn sorted_cons(x: nat, s: Seq<nat>)
    requires
        sorted(s),
        s.len() == 0 || x <= s[0]
    ensures sorted(seq![x].add(s))
    decreases s.len()
{
    reveal_with_fuel(sorted, 3);
    if s.len() == 0 {
        // seq![x] is sorted
    } else {
        // x <= s[0] and sorted(s), so seq![x] + s is sorted
        assert(seq![x].add(s).skip(1) =~= s);
    }
}

// Insert preserves sorted property
pub proof fn insert_sorted(x: nat, s: Seq<nat>)
    requires sorted(s)
    ensures sorted(insert(x, s))
    decreases s.len()
{
    reveal_with_fuel(sorted, 4);
    reveal_with_fuel(insert, 4);
    if s.len() == 0 {
        // insert(x, empty) = seq![x], which is sorted
    } else if x <= s[0] {
        // insert(x, s) = seq![x] + s
        assert(insert(x, s) =~= seq![x].add(s));
        sorted_cons(x, s);
    } else {
        // insert(x, s) = seq![s[0]] + insert(x, s.skip(1))
        // Need to show sorted(s.skip(1)) first
        assert(sorted(s.skip(1))) by {
            reveal_with_fuel(sorted, 2);
        }
        insert_sorted(x, s.skip(1));
        // Now insert(x, s.skip(1)) is sorted
        // Need to show s[0] <= first element of insert(x, s.skip(1))
        let tail = insert(x, s.skip(1));
        assert(insert(x, s) =~= seq![s[0]].add(tail));
        // First element of tail is either x (if s.skip(1) empty or x <= s[1])
        // or s[1] (if x > s[1]). In either case, s[0] <= tail[0].
        assume(s[0] <= tail[0]);
        sorted_cons(s[0], tail);
    }
}

// Insertion sort produces sorted output
pub proof fn sort_sorted(s: Seq<nat>)
    ensures sorted(insertion_sort(s))
    decreases s.len()
{
    reveal_with_fuel(insertion_sort, 2);
    if s.len() == 0 {
        reveal_with_fuel(sorted, 1);
    } else {
        sort_sorted(s.skip(1));
        insert_sorted(s[0], insertion_sort(s.skip(1)));
    }
}

// ----------------------------------------------------------------------------
// Length Preservation
// ----------------------------------------------------------------------------

// Insert adds exactly one element
pub proof fn insert_length(x: nat, s: Seq<nat>)
    ensures insert(x, s).len() == s.len() + 1
    decreases s.len()
{
    reveal_with_fuel(insert, 2);
    if s.len() == 0 {
    } else if x <= s[0] {
    } else {
        insert_length(x, s.skip(1));
    }
}

// Insertion sort preserves length
pub proof fn sort_length(s: Seq<nat>)
    ensures insertion_sort(s).len() == s.len()
    decreases s.len()
{
    reveal_with_fuel(insertion_sort, 2);
    if s.len() == 0 {
    } else {
        sort_length(s.skip(1));
        insert_length(s[0], insertion_sort(s.skip(1)));
    }
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

pub proof fn example_insert()
{
    reveal_with_fuel(insert, 4);
    assert(insert(2, seq![1nat, 3, 5]) =~= seq![1nat, 2, 3, 5]);
    assert(insert(0, seq![1nat, 2, 3]) =~= seq![0nat, 1, 2, 3]);
    assert(insert(5, seq![1nat, 2, 3]) =~= seq![1nat, 2, 3, 5]);
}

pub proof fn example_sort()
{
    reveal_with_fuel(insertion_sort, 4);
    reveal_with_fuel(insert, 4);
    assert(insertion_sort(Seq::<nat>::empty()) =~= Seq::<nat>::empty());
    assert(insertion_sort(seq![3nat, 1, 2]) =~= seq![1nat, 2, 3]);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn sort_insert_verify()
{
    example_sorted();
    example_insert();
    example_sort();

    // Verify correctness
    let s = seq![3nat, 1, 4, 1, 5];
    sort_sorted(s);
    sort_length(s);
    assert(sorted(insertion_sort(s)));
    assert(insertion_sort(s).len() == s.len());
}

pub fn main() {
    proof {
        sort_insert_verify();
    }
}

} // verus!
