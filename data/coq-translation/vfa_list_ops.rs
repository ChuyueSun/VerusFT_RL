use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: List Operations (Supporting VFA)
// ============================================================================

pub enum List { Nil, Cons { head: nat, tail: Box<List> } }

pub open spec fn len(l: List) -> nat decreases l {
    match l { List::Nil => 0, List::Cons { tail, .. } => 1 + len(*tail) }
}

pub open spec fn nth(l: List, n: nat) -> Option<nat> decreases l, n {
    match l {
        List::Nil => None,
        List::Cons { head, tail } => if n == 0 { Some(head) } else { nth(*tail, (n-1) as nat) }
    }
}

pub open spec fn take(l: List, n: nat) -> List decreases l, n {
    if n == 0 { List::Nil }
    else { match l {
        List::Nil => List::Nil,
        List::Cons { head, tail } => List::Cons { head, tail: Box::new(take(*tail, (n-1) as nat)) }
    }}
}

pub open spec fn drop(l: List, n: nat) -> List decreases l, n {
    if n == 0 { l }
    else { match l {
        List::Nil => List::Nil,
        List::Cons { tail, .. } => drop(*tail, (n-1) as nat)
    }}
}

pub open spec fn filter(l: List, p: spec_fn(nat) -> bool) -> List decreases l {
    match l {
        List::Nil => List::Nil,
        List::Cons { head, tail } =>
            if p(head) { List::Cons { head, tail: Box::new(filter(*tail, p)) } }
            else { filter(*tail, p) }
    }
}

pub open spec fn map(l: List, f: spec_fn(nat) -> nat) -> List decreases l {
    match l {
        List::Nil => List::Nil,
        List::Cons { head, tail } => List::Cons { head: f(head), tail: Box::new(map(*tail, f)) }
    }
}

pub proof fn map_len(l: List, f: spec_fn(nat) -> nat)
    ensures len(map(l, f)) == len(l) decreases l
{
    reveal_with_fuel(len, 2); reveal_with_fuel(map, 2);
    match l { List::Nil => {} List::Cons { tail, .. } => { map_len(*tail, f); } }
}

pub proof fn example_list_ops() {
    reveal_with_fuel(len, 3);
    reveal_with_fuel(nth, 3);
    let l = List::Cons { head: 1, tail: Box::new(List::Cons { head: 2, tail: Box::new(List::Nil) }) };
    assert(len(l) == 2);
    assert(nth(l, 0) == Some(1nat));
    assert(nth(l, 1) == Some(2nat));
}

pub proof fn list_ops_verify() { example_list_ops(); }
pub fn main() { proof { list_ops_verify(); } }

} // verus!
