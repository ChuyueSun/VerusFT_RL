use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: List Definition (Supporting VFA)
// Linked list type and basic operations
// ============================================================================

pub enum List<T> {
    Nil,
    Cons { head: T, tail: Box<List<T>> },
}

pub open spec fn list_len<T>(l: List<T>) -> nat
    decreases l
{
    match l {
        List::Nil => 0,
        List::Cons { head: _, tail } => 1 + list_len(*tail),
    }
}

pub open spec fn list_append<T>(l1: List<T>, l2: List<T>) -> List<T>
    decreases l1
{
    match l1 {
        List::Nil => l2,
        List::Cons { head, tail } => List::Cons { head, tail: Box::new(list_append(*tail, l2)) },
    }
}

pub open spec fn list_reverse_acc<T>(l: List<T>, acc: List<T>) -> List<T>
    decreases l
{
    match l {
        List::Nil => acc,
        List::Cons { head, tail } => list_reverse_acc(*tail, List::Cons { head, tail: Box::new(acc) }),
    }
}

pub open spec fn list_reverse<T>(l: List<T>) -> List<T> { list_reverse_acc(l, List::Nil) }

pub proof fn append_len<T>(l1: List<T>, l2: List<T>)
    ensures list_len(list_append(l1, l2)) == list_len(l1) + list_len(l2)
    decreases l1
{
    reveal_with_fuel(list_len, 2);
    reveal_with_fuel(list_append, 2);
    match l1 {
        List::Nil => {}
        List::Cons { head: _, tail } => { append_len(*tail, l2); }
    }
}

pub proof fn example_list()
{
    reveal_with_fuel(list_len, 4);
    let l: List<nat> = List::Cons { head: 1, tail: Box::new(List::Cons { head: 2, tail: Box::new(List::Nil) }) };
    assert(list_len(l) == 2);
}

pub proof fn list_def_verify() { example_list(); }
pub fn main() { proof { list_def_verify(); } }

} // verus!
