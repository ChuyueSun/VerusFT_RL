use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Double-Ended Queue (Supporting VFA)
// ============================================================================

pub struct Deque { pub front: Seq<nat>, pub back: Seq<nat> }

pub open spec fn deque_empty() -> Deque { Deque { front: Seq::empty(), back: Seq::empty() } }
pub open spec fn deque_is_empty(d: Deque) -> bool { d.front.len() == 0 && d.back.len() == 0 }
pub open spec fn deque_len(d: Deque) -> nat { d.front.len() + d.back.len() }

pub open spec fn push_front(x: nat, d: Deque) -> Deque { Deque { front: seq![x] + d.front, back: d.back } }
pub open spec fn push_back(d: Deque, x: nat) -> Deque { Deque { front: d.front, back: d.back.push(x) } }

pub open spec fn pop_front(d: Deque) -> (Option<nat>, Deque) {
    if d.front.len() > 0 { (Some(d.front[0]), Deque { front: d.front.skip(1), back: d.back }) }
    else if d.back.len() > 0 { (Some(d.back[d.back.len() - 1]), Deque { front: Seq::empty(), back: d.back.take((d.back.len() - 1) as int) }) }
    else { (None, d) }
}

pub proof fn push_front_len(x: nat, d: Deque) ensures deque_len(push_front(x, d)) == deque_len(d) + 1 {}
pub proof fn push_back_len(d: Deque, x: nat) ensures deque_len(push_back(d, x)) == deque_len(d) + 1 {}

pub proof fn example_deque() {
    let d = deque_empty();
    assert(deque_is_empty(d));
    let d1 = push_front(5, d);
    push_front_len(5, d);
    assert(deque_len(d1) == 1);
}

pub proof fn deque_def_verify() { example_deque(); }
pub fn main() { proof { deque_def_verify(); } }

} // verus!
