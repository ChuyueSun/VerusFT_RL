use vstd::prelude::*;

verus! {

// Rel (Software Foundations, LF/Rel) style snippets in Verus.
//
// Focus: inductively defined relations and (reflexive) transitive closure.

pub type Key = nat;

// A simple step relation: one-step successor.
pub open spec fn step1(x: nat, y: nat) -> bool {
    y == x + 1
}

// Reflexive-transitive closure of `step1`, represented as a "snoc" chain.
//
// `Refl{x}`: x reaches x.
// `Snoc{prev, y}`: extend an existing path `prev` by one final step to y.
//
// Invariant `rt_inv(e)` captures the intended meaning.

pub enum RtStep1 {
    Refl { x: nat },
    Snoc { prev: Box<RtStep1>, y: nat },
}

pub open spec fn rt_lhs(e: RtStep1) -> nat
    decreases e
{
    match e {
        RtStep1::Refl { x } => x,
        RtStep1::Snoc { prev, y: _ } => rt_lhs(*prev),
    }
}

pub open spec fn rt_rhs(e: RtStep1) -> nat
    decreases e
{
    match e {
        RtStep1::Refl { x } => x,
        RtStep1::Snoc { prev: _, y } => y,
    }
}

pub open spec fn rt_inv(e: RtStep1) -> bool
    decreases e
{
    match e {
        RtStep1::Refl { .. } => true,
        RtStep1::Snoc { prev, y } => rt_inv(*prev) && step1(rt_rhs(*prev), y),
    }
}

pub proof fn lemma_rt_refl(x: nat) -> (e: RtStep1)
    ensures rt_inv(e),
        rt_lhs(e) == x,
        rt_rhs(e) == x
{
    RtStep1::Refl { x }
}

pub proof fn lemma_rt_snoc(prev: RtStep1, y: nat) -> (e: RtStep1)
    requires rt_inv(prev),
        step1(rt_rhs(prev), y)
    ensures rt_inv(e),
        rt_lhs(e) == rt_lhs(prev),
        rt_rhs(e) == y
{
    RtStep1::Snoc { prev: Box::new(prev), y }
}

pub proof fn lemma_rt_inv_unfold(e: RtStep1)
    ensures rt_inv(e) == match e {
        RtStep1::Refl { .. } => true,
        RtStep1::Snoc { prev, y } => rt_inv(*prev) && step1(rt_rhs(*prev), y),
    }
{
    reveal_with_fuel(rt_inv, 1);
}

pub proof fn lemma_rt_lhs_unfold(e: RtStep1)
    ensures rt_lhs(e) == match e {
        RtStep1::Refl { x } => x,
        RtStep1::Snoc { prev, y: _ } => rt_lhs(*prev),
    }
{
    reveal_with_fuel(rt_lhs, 1);
}

pub proof fn lemma_rt_rhs_unfold(e: RtStep1)
    ensures rt_rhs(e) == match e {
        RtStep1::Refl { x } => x,
        RtStep1::Snoc { prev: _, y } => y,
    }
{
    reveal_with_fuel(rt_rhs, 1);
}

// Concatenation (transitivity): if x~*y and y~*z then x~*z.
pub proof fn lemma_rt_trans(e1: RtStep1, e2: RtStep1) -> (e: RtStep1)
    requires rt_inv(e1),
        rt_inv(e2),
        rt_rhs(e1) == rt_lhs(e2)
    ensures rt_inv(e),
        rt_lhs(e) == rt_lhs(e1),
        rt_rhs(e) == rt_rhs(e2)
    decreases e2
{
    match e2 {
        RtStep1::Refl { x: _ } => {
            // e2 is reflexive at rt_lhs(e2)=rt_rhs(e2), so result is e1.
            e1
        }
        RtStep1::Snoc { prev, y } => {
            // First, concatenate e1 with the prefix.
            let prev_e2 = *prev;
            let mid = lemma_rt_trans(e1, prev_e2);

            // Now extend with the last step.
            // From rt_inv(e2): step1(rt_rhs(prev), y)
            // and from the recursion: rt_rhs(mid) == rt_rhs(prev).
            assert(step1(rt_rhs(mid), y));
            lemma_rt_snoc(mid, y)
        }
    }
}

// Closure implies ordering: any step1-chain from x to y means x <= y.
pub proof fn lemma_rt_implies_le(e: RtStep1)
    requires rt_inv(e)
    ensures rt_lhs(e) <= rt_rhs(e)
    decreases e
{
    match e {
        RtStep1::Refl { x: _ } => {
        }
        RtStep1::Snoc { prev, y } => {
            let p = *prev;
            lemma_rt_implies_le(p);
            // rt_inv(e) implies step1(rt_rhs(p), y), hence y = rt_rhs(p) + 1.
            assert(step1(rt_rhs(p), y));
            assert(rt_rhs(p) < y);
            // Combine: rt_lhs(p) <= rt_rhs(p) < y, and rt_lhs(e)=rt_lhs(p), rt_rhs(e)=y.
            assert(rt_lhs(p) <= y);
        }
    }
}

// Ordering implies closure: if x <= y then there is a step1-chain from x to y.
pub proof fn lemma_le_implies_rt(x: nat, y: nat) -> (e: RtStep1)
    requires x <= y
    ensures rt_inv(e),
        rt_lhs(e) == x,
        rt_rhs(e) == y
    decreases y
{
    if x == y {
        lemma_rt_refl(x)
    } else {
        // From x <= y and x != y we get x < y, so y > 0 and x <= y-1.
        assert(x < y);
        assert(y > 0);
        let y1 = (y - 1) as nat;
        assert(x <= y1);

        let prev = lemma_le_implies_rt(x, y1);
        // Extend by a final successor step from y-1 to y.
        assert(step1(y1, y));
        lemma_rt_snoc(prev, y)
    }
}

// A tiny example: 2 ~* 5 via three successor steps.
pub proof fn ex_chain_2_to_5() -> (e: RtStep1)
    ensures rt_inv(e),
        rt_lhs(e) == 2,
        rt_rhs(e) == 5
{
    let e2 = lemma_rt_refl(2);
    let e3 = lemma_rt_snoc(e2, 3);
    let e4 = lemma_rt_snoc(e3, 4);
    lemma_rt_snoc(e4, 5)
}

// And the corresponding inequality fact.
pub proof fn ex_chain_2_to_5_implies_le()
{
    let e = ex_chain_2_to_5();
    lemma_rt_implies_le(e);
}

pub fn main() {
    proof {
        let _e = ex_chain_2_to_5();
        ex_chain_2_to_5_implies_le();

        // Round-trip: x<=y -> closure -> x<=y.
        let e = lemma_le_implies_rt(4, 9);
        lemma_rt_implies_le(e);

        // Transitivity demo.
        let e1 = lemma_le_implies_rt(1, 3);
        let e2 = lemma_le_implies_rt(3, 6);
        let e3 = lemma_rt_trans(e1, e2);
        lemma_rt_implies_le(e3);
    }
}

} // verus!
