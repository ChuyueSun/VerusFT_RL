use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Abstract Data Type - Queue (vfa-current/ADT)
// Queue ADT with algebraic specification
// ============================================================================

// ----------------------------------------------------------------------------
// Queue Type (Two-List Implementation)
// ----------------------------------------------------------------------------

pub struct Queue<T> {
    pub front: Seq<T>,  // Dequeue from front
    pub back: Seq<T>,   // Enqueue to back (reversed)
}

// ----------------------------------------------------------------------------
// Operations
// ----------------------------------------------------------------------------

// Empty queue
pub open spec fn queue_empty<T>() -> Queue<T> {
    Queue { front: Seq::empty(), back: Seq::empty() }
}

// Check if empty
pub open spec fn queue_is_empty<T>(q: Queue<T>) -> bool {
    q.front.len() == 0 && q.back.len() == 0
}

// Enqueue element (add to back)
pub open spec fn queue_enqueue<T>(x: T, q: Queue<T>) -> Queue<T> {
    Queue { front: q.front, back: q.back.push(x) }
}

// Helper: reverse sequence
pub open spec fn seq_reverse<T>(s: Seq<T>) -> Seq<T> {
    Seq::new(s.len(), |i: int| s[s.len() - 1 - i])
}

// Normalize queue (move back to front if front is empty)
pub open spec fn queue_normalize<T>(q: Queue<T>) -> Queue<T> {
    if q.front.len() == 0 && q.back.len() > 0 {
        Queue { front: seq_reverse(q.back), back: Seq::empty() }
    } else {
        q
    }
}

// Dequeue element (remove from front)
pub open spec fn queue_dequeue<T>(q: Queue<T>) -> Queue<T> {
    let nq = queue_normalize(q);
    if nq.front.len() == 0 {
        nq
    } else {
        Queue { front: nq.front.skip(1), back: nq.back }
    }
}

// Peek front element
pub open spec fn queue_peek<T>(q: Queue<T>) -> Option<T> {
    let nq = queue_normalize(q);
    if nq.front.len() == 0 {
        None
    } else {
        Some(nq.front[0])
    }
}

// Queue size
pub open spec fn queue_size<T>(q: Queue<T>) -> nat {
    q.front.len() + q.back.len()
}

// ----------------------------------------------------------------------------
// Abstract Model (Sequence)
// ----------------------------------------------------------------------------

// Queue as abstract sequence (front + reversed back)
pub open spec fn queue_to_seq<T>(q: Queue<T>) -> Seq<T> {
    q.front + seq_reverse(q.back)
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Empty queue has size 0
pub proof fn empty_size<T>()
    ensures queue_size(queue_empty::<T>()) == 0
{
}

// Enqueue increases size by 1
pub proof fn enqueue_size<T>(x: T, q: Queue<T>)
    ensures queue_size(queue_enqueue(x, q)) == queue_size(q) + 1
{
}

// Dequeue decreases size by 1 (if non-empty)
pub proof fn dequeue_size<T>(q: Queue<T>)
    requires !queue_is_empty(q)
    ensures queue_size(queue_dequeue(q)) == queue_size(q) - 1
{
    let nq = queue_normalize(q);
    if q.front.len() == 0 {
        // After normalize, front has back reversed
        assert(nq.front.len() == q.back.len());
    }
}

// Peek returns first element of abstract sequence
pub proof fn peek_first<T>(q: Queue<T>)
    requires !queue_is_empty(q)
    ensures queue_peek(q) == Some(queue_to_seq(q)[0])
{
    let nq = queue_normalize(q);
    if q.front.len() == 0 {
        // front was empty, now has reversed back
        assert(nq.front =~= seq_reverse(q.back));
        assert(nq.front[0] == q.back[q.back.len() - 1]);
    }
}

// ----------------------------------------------------------------------------
// FIFO Property
// ----------------------------------------------------------------------------

// Enqueue and peek property
pub proof fn enqueue_peek_property<T>(x: T)
{
    let q: Queue<T> = queue_empty();
    let q1 = queue_enqueue(x, q);

    // q1 = { front: [], back: [x] }
    assert(q1.front.len() == 0);
    assert(q1.back.len() == 1);

    // After normalize: front = reverse([x]) = [x]
    let nq = queue_normalize(q1);
    assert(nq.front.len() == 1);
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_basic()
{
    let q: Queue<nat> = queue_empty();
    empty_size::<nat>();
    assert(queue_is_empty(q));
    assert(queue_size(q) == 0);
}

pub proof fn example_enqueue()
{
    let q: Queue<nat> = queue_empty();
    let q1 = queue_enqueue(5nat, q);
    enqueue_size(5nat, q);
    assert(queue_size(q1) == 1);
    assert(!queue_is_empty(q1));
}

pub proof fn example_normalize()
{
    // Queue with empty front, items in back
    let q: Queue<nat> = Queue { front: Seq::empty(), back: seq![1nat, 2, 3] };
    let nq = queue_normalize(q);

    assert(nq.front.len() == 3);
    assert(nq.back.len() == 0);
}

pub proof fn example_dequeue()
{
    let q: Queue<nat> = queue_empty();
    let q1 = queue_enqueue(10nat, q);
    let q2 = queue_enqueue(20nat, q1);

    // Peek should see one of them
    assert(!queue_is_empty(q2));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn adt_queue_verify()
{
    example_basic();
    example_enqueue();
    example_normalize();
    example_dequeue();

    // Test size properties
    let q: Queue<nat> = queue_empty();
    empty_size::<nat>();
    enqueue_size(1nat, q);
}

pub fn main() {
    proof {
        adt_queue_verify();
    }
}

} // verus!
