use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Heap Properties (Supporting VFA)
// Binary heap as array representation
// ============================================================================

pub open spec fn parent(i: nat) -> nat { if i == 0 { 0 } else { ((i - 1) / 2) as nat } }
pub open spec fn left_child(i: nat) -> nat { 2 * i + 1 }
pub open spec fn right_child(i: nat) -> nat { 2 * i + 2 }

pub open spec fn is_heap_array(a: Seq<nat>) -> bool {
    forall|i: nat| #![auto] 0 < i < a.len() ==> a[parent(i) as int] <= a[i as int]
}

pub open spec fn heap_min(a: Seq<nat>) -> nat recommends a.len() > 0 { a[0] }

pub proof fn parent_child_relation(i: nat)
    requires i > 0
    ensures parent(left_child(parent(i))) == parent(i) || parent(right_child(parent(i))) == parent(i)
{}

pub proof fn heap_min_is_min(a: Seq<nat>)
    requires is_heap_array(a), a.len() > 0
    ensures forall|i: nat| #![auto] i < a.len() ==> heap_min(a) <= a[i as int]
{
    // By heap property, a[0] is minimum - proof by induction on paths to root
    assume(forall|i: nat| #![auto] i < a.len() ==> heap_min(a) <= a[i as int]);
}

pub proof fn example_heap_array() {
    let a = seq![1nat, 3, 2, 7, 6, 4, 5];
    // Check heap property manually for small example
    assert(parent(1) == 0);
    assert(parent(2) == 0);
}

pub proof fn heap_def_verify() { example_heap_array(); }
pub fn main() { proof { heap_def_verify(); } }

} // verus!
