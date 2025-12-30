use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Abstract Data Type - Stack (vfa-current/ADT)
// Stack ADT with algebraic specification
// ============================================================================

// ----------------------------------------------------------------------------
// Stack Type
// ----------------------------------------------------------------------------

pub enum Stack<T> {
    Empty,
    Push { top: T, rest: Box<Stack<T>> },
}

// ----------------------------------------------------------------------------
// Operations
// ----------------------------------------------------------------------------

// Empty stack
pub open spec fn stack_empty<T>() -> Stack<T> {
    Stack::Empty
}

// Check if empty
pub open spec fn stack_is_empty<T>(s: Stack<T>) -> bool {
    matches!(s, Stack::Empty)
}

// Push element
pub open spec fn stack_push<T>(x: T, s: Stack<T>) -> Stack<T> {
    Stack::Push { top: x, rest: Box::new(s) }
}

// Pop element (returns rest of stack)
pub open spec fn stack_pop<T>(s: Stack<T>) -> Stack<T> {
    match s {
        Stack::Empty => Stack::Empty,
        Stack::Push { top: _, rest } => *rest,
    }
}

// Peek top element
pub open spec fn stack_peek<T>(s: Stack<T>) -> Option<T> {
    match s {
        Stack::Empty => None,
        Stack::Push { top, rest: _ } => Some(top),
    }
}

// Stack size
pub open spec fn stack_size<T>(s: Stack<T>) -> nat
    decreases s
{
    match s {
        Stack::Empty => 0,
        Stack::Push { top: _, rest } => 1 + stack_size(*rest),
    }
}

// ----------------------------------------------------------------------------
// Algebraic Laws
// ----------------------------------------------------------------------------

// Pop after push returns original stack
pub proof fn pop_push<T>(x: T, s: Stack<T>)
    ensures stack_pop(stack_push(x, s)) == s
{
}

// Peek after push returns pushed element
pub proof fn peek_push<T>(x: T, s: Stack<T>)
    ensures stack_peek(stack_push(x, s)) == Some(x)
{
}

// Push makes stack non-empty
pub proof fn push_not_empty<T>(x: T, s: Stack<T>)
    ensures !stack_is_empty(stack_push(x, s))
{
}

// Empty stack is empty
pub proof fn empty_is_empty<T>()
    ensures stack_is_empty(stack_empty::<T>())
{
}

// Size after push
pub proof fn size_push<T>(x: T, s: Stack<T>)
    ensures stack_size(stack_push(x, s)) == stack_size(s) + 1
{
}

// Size of empty
pub proof fn size_empty<T>()
    ensures stack_size(stack_empty::<T>()) == 0
{
}

// ----------------------------------------------------------------------------
// Stack from Sequence
// ----------------------------------------------------------------------------

pub open spec fn stack_from_seq<T>(s: Seq<T>) -> Stack<T>
    decreases s.len()
{
    if s.len() == 0 {
        Stack::Empty
    } else {
        stack_push(s[s.len() - 1], stack_from_seq(s.take(s.len() - 1)))
    }
}

pub open spec fn stack_to_seq<T>(s: Stack<T>) -> Seq<T>
    decreases s
{
    match s {
        Stack::Empty => Seq::empty(),
        Stack::Push { top, rest } => stack_to_seq(*rest).push(top),
    }
}

// Round-trip property
pub proof fn stack_seq_roundtrip<T>(s: Seq<T>)
    ensures stack_to_seq(stack_from_seq(s)) =~= s
    decreases s.len()
{
    reveal_with_fuel(stack_from_seq, 2);
    reveal_with_fuel(stack_to_seq, 2);
    if s.len() > 0 {
        stack_seq_roundtrip(s.take(s.len() - 1));
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_basic_ops()
{
    let s: Stack<nat> = stack_empty();
    empty_is_empty::<nat>();
    assert(stack_is_empty(s));

    let s1 = stack_push(5nat, s);
    push_not_empty(5nat, s);
    assert(!stack_is_empty(s1));

    peek_push(5nat, s);
    assert(stack_peek(s1) == Some(5nat));

    pop_push(5nat, s);
    assert(stack_pop(s1) == s);
}

pub proof fn example_size()
{
    reveal_with_fuel(stack_size, 4);

    let s: Stack<nat> = stack_empty();
    size_empty::<nat>();
    assert(stack_size(s) == 0);

    let s1 = stack_push(1nat, s);
    size_push(1nat, s);
    assert(stack_size(s1) == 1);

    let s2 = stack_push(2nat, s1);
    size_push(2nat, s1);
    assert(stack_size(s2) == 2);
}

pub proof fn example_lifo()
{
    let s: Stack<nat> = stack_empty();
    let s1 = stack_push(1nat, s);
    let s2 = stack_push(2nat, s1);
    let s3 = stack_push(3nat, s2);

    // LIFO order
    assert(stack_peek(s3) == Some(3nat));
    assert(stack_peek(stack_pop(s3)) == Some(2nat));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn adt_stack_verify()
{
    example_basic_ops();
    example_size();
    example_lifo();

    // Test laws
    let s: Stack<nat> = stack_empty();
    pop_push(42nat, s);
    peek_push(42nat, s);
}

pub fn main() {
    proof {
        adt_stack_verify();
    }
}

} // verus!
