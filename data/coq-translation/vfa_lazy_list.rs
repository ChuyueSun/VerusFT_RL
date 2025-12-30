use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Lazy List (Supporting VFA)
// ============================================================================

pub enum LazyList { Nil, Cons { head: nat, tail: Box<LazyList> } }

pub open spec fn take(l: LazyList, n: nat) -> Seq<nat> decreases n, l {
    if n == 0 { Seq::empty() }
    else { match l { LazyList::Nil => Seq::empty(), LazyList::Cons { head, tail } => seq![head] + take(*tail, (n-1) as nat) } }
}

pub open spec fn iterate(f: spec_fn(nat) -> nat, start: nat, n: nat) -> Seq<nat> decreases n {
    if n == 0 { Seq::empty() } else { seq![start] + iterate(f, f(start), (n-1) as nat) }
}

pub open spec fn repeat(x: nat, n: nat) -> Seq<nat> { Seq::new(n, |_i: int| x) }

pub proof fn take_len(l: LazyList, n: nat) ensures take(l, n).len() <= n decreases n, l {
    reveal_with_fuel(take, 2);
    if n > 0 { match l { LazyList::Nil => {} LazyList::Cons { tail, .. } => { take_len(*tail, (n-1) as nat); } } }
}

pub proof fn example_lazy() { reveal_with_fuel(take, 3); reveal_with_fuel(iterate, 3); }
pub proof fn lazy_list_verify() { example_lazy(); }
pub fn main() { proof { lazy_list_verify(); } }

} // verus!
