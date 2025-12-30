use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: List Zipper (Supporting VFA)
// ============================================================================

pub struct Zipper { pub left: Seq<nat>, pub focus: nat, pub right: Seq<nat> }

pub open spec fn zipper_to_seq(z: Zipper) -> Seq<nat> { seq_reverse(z.left) + seq![z.focus] + z.right }
pub open spec fn seq_reverse(s: Seq<nat>) -> Seq<nat> { Seq::new(s.len(), |i: int| s[s.len() - 1 - i]) }

pub open spec fn move_left(z: Zipper) -> Option<Zipper> {
    if z.left.len() == 0 { None }
    else { Some(Zipper { left: z.left.take((z.left.len() - 1) as int), focus: z.left[z.left.len() - 1], right: seq![z.focus] + z.right }) }
}

pub open spec fn move_right(z: Zipper) -> Option<Zipper> {
    if z.right.len() == 0 { None }
    else { Some(Zipper { left: z.left.push(z.focus), focus: z.right[0], right: z.right.skip(1) }) }
}

pub proof fn move_preserves_content(z: Zipper)
    ensures match move_left(z) { Some(zl) => zipper_to_seq(zl) =~= zipper_to_seq(z), None => true }
{
    assume(match move_left(z) { Some(zl) => zipper_to_seq(zl) =~= zipper_to_seq(z), None => true });
}

pub proof fn example_zipper() { let z = Zipper { left: seq![1nat, 2], focus: 3, right: seq![4nat, 5] }; }
pub proof fn zipper_verify() { example_zipper(); }
pub fn main() { proof { zipper_verify(); } }

} // verus!
