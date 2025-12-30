use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Ring Buffer (Supporting VFA)
// ============================================================================

pub struct RingBuffer { pub data: Seq<nat>, pub head: nat, pub len: nat, pub cap: nat }

pub open spec fn rb_valid(rb: RingBuffer) -> bool {
    rb.cap > 0 && rb.data.len() == rb.cap && rb.len <= rb.cap && rb.head < rb.cap
}

pub open spec fn rb_empty(cap: nat) -> RingBuffer recommends cap > 0 {
    RingBuffer { data: Seq::new(cap, |_i: int| 0nat), head: 0, len: 0, cap }
}

pub open spec fn rb_is_empty(rb: RingBuffer) -> bool { rb.len == 0 }
pub open spec fn rb_is_full(rb: RingBuffer) -> bool { rb.len == rb.cap }

pub open spec fn rb_index(rb: RingBuffer, i: nat) -> nat recommends i < rb.len {
    (rb.head + i) % rb.cap
}

pub open spec fn rb_get(rb: RingBuffer, i: nat) -> nat recommends rb_valid(rb), i < rb.len {
    rb.data[rb_index(rb, i) as int]
}

pub proof fn rb_empty_valid(cap: nat) requires cap > 0 ensures rb_valid(rb_empty(cap)) {}
pub proof fn rb_empty_is_empty(cap: nat) requires cap > 0 ensures rb_is_empty(rb_empty(cap)) {}

pub proof fn example_ring_buffer() {
    rb_empty_valid(10);
    rb_empty_is_empty(10);
    let rb = rb_empty(5);
    assert(rb_is_empty(rb));
    assert(!rb_is_full(rb));
}

pub proof fn ring_buffer_verify() { example_ring_buffer(); }
pub fn main() { proof { ring_buffer_verify(); } }

} // verus!
