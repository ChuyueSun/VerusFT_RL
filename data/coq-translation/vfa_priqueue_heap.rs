use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Priority Queue Heap (vfa-current/Priqueue)
// Heap-based priority queue implementation
// ============================================================================

// ----------------------------------------------------------------------------
// Heap Type (Binary Heap as Tree)
// ----------------------------------------------------------------------------

pub enum Heap {
    E,  // Empty
    T { priority: nat, left: Box<Heap>, right: Box<Heap> },
}

// ----------------------------------------------------------------------------
// Heap Invariant (Min-Heap Property)
// ----------------------------------------------------------------------------

// All nodes have priority >= bound
pub open spec fn all_ge(h: Heap, bound: nat) -> bool
    decreases h
{
    match h {
        Heap::E => true,
        Heap::T { priority, left, right } =>
            priority >= bound &&
            all_ge(*left, bound) &&
            all_ge(*right, bound),
    }
}

// Min-heap property: each node's priority <= its children's priorities
pub open spec fn is_heap(h: Heap) -> bool
    decreases h
{
    match h {
        Heap::E => true,
        Heap::T { priority, left, right } =>
            all_ge(*left, priority) &&
            all_ge(*right, priority) &&
            is_heap(*left) &&
            is_heap(*right),
    }
}

// ----------------------------------------------------------------------------
// Operations
// ----------------------------------------------------------------------------

// Empty heap
pub open spec fn heap_empty() -> Heap {
    Heap::E
}

// Check if empty
pub open spec fn heap_is_empty(h: Heap) -> bool {
    matches!(h, Heap::E)
}

// Get minimum (root)
pub open spec fn heap_find_min(h: Heap) -> Option<nat> {
    match h {
        Heap::E => None,
        Heap::T { priority, left: _, right: _ } => Some(priority),
    }
}

// Merge two heaps
pub open spec fn heap_merge(h1: Heap, h2: Heap) -> Heap
    decreases h1, h2
{
    match (h1, h2) {
        (Heap::E, h) => h,
        (h, Heap::E) => h,
        (Heap::T { priority: p1, left: l1, right: r1 },
         Heap::T { priority: p2, left: l2, right: r2 }) =>
            if p1 <= p2 {
                Heap::T {
                    priority: p1,
                    left: Box::new(heap_merge(*r1, h2)),
                    right: l1,
                }
            } else {
                Heap::T {
                    priority: p2,
                    left: Box::new(heap_merge(h1, *r2)),
                    right: l2,
                }
            }
    }
}

// Insert element
pub open spec fn heap_insert(x: nat, h: Heap) -> Heap {
    heap_merge(Heap::T { priority: x, left: Box::new(Heap::E), right: Box::new(Heap::E) }, h)
}

// Delete minimum
pub open spec fn heap_delete_min(h: Heap) -> Heap {
    match h {
        Heap::E => Heap::E,
        Heap::T { priority: _, left, right } => heap_merge(*left, *right),
    }
}

// ----------------------------------------------------------------------------
// Size
// ----------------------------------------------------------------------------

pub open spec fn heap_size(h: Heap) -> nat
    decreases h
{
    match h {
        Heap::E => 0,
        Heap::T { priority: _, left, right } =>
            1 + heap_size(*left) + heap_size(*right),
    }
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Empty heap is valid
pub proof fn heap_empty_valid()
    ensures is_heap(heap_empty())
{
}

// Find min returns minimum element
pub proof fn heap_find_min_is_min(h: Heap)
    requires is_heap(h), !heap_is_empty(h)
    ensures match heap_find_min(h) {
        Some(m) => all_ge(h, m),
        None => false,
    }
{
    reveal_with_fuel(all_ge, 2);
    reveal_with_fuel(is_heap, 2);
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_empty_heap()
{
    let h = heap_empty();
    heap_empty_valid();
    assert(is_heap(h));
    assert(heap_is_empty(h));
    assert(heap_find_min(h).is_none());
}

pub proof fn example_singleton()
{
    reveal_with_fuel(is_heap, 3);
    reveal_with_fuel(all_ge, 3);
    reveal_with_fuel(heap_merge, 3);

    let h = heap_insert(42, heap_empty());
    // h should be T { 42, E, E }
    assert(heap_find_min(h) == Some(42nat));
}

pub proof fn example_insert_two()
{
    reveal_with_fuel(is_heap, 4);
    reveal_with_fuel(all_ge, 4);
    reveal_with_fuel(heap_merge, 4);

    let h1 = heap_insert(10, heap_empty());
    let h2 = heap_insert(5, h1);

    // Minimum should be 5
    assert(heap_find_min(h2) == Some(5nat));
}

pub proof fn example_delete_min()
{
    reveal_with_fuel(heap_merge, 4);

    let h1 = heap_insert(10, heap_empty());
    let h2 = heap_insert(5, h1);
    let h3 = heap_delete_min(h2);

    // After deleting 5, min should be 10
    assert(heap_find_min(h3) == Some(10nat));
}

pub proof fn example_size()
{
    reveal_with_fuel(heap_size, 3);
    reveal_with_fuel(heap_merge, 3);

    let h = heap_empty();
    assert(heap_size(h) == 0);

    let h1 = heap_insert(5, h);
    assert(heap_size(h1) == 1);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn priqueue_heap_verify()
{
    example_empty_heap();
    example_singleton();
    example_insert_two();
    example_delete_min();
    example_size();
}

pub fn main() {
    proof {
        priqueue_heap_verify();
    }
}

} // verus!
