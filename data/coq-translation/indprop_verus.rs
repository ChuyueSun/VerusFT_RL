use vstd::prelude::*;
use vstd::seq_lib::*;
use vstd::seq::group_seq_axioms;

verus! {

broadcast use group_seq_axioms;

// IndProp (Software Foundations, LF/IndProp) style snippets in Verus.
// We encode inductive propositions using *evidence objects* (enums) whose
// constructors correspond to the inductive rules.

// ------------------------
// Even (inductive)
// ------------------------

// A simple computational predicate, used for soundness/completeness statements.
pub open spec fn is_even(n: nat) -> bool
    decreases n
{
    if n == 0 {
        true
    } else if n == 1 {
        false
    } else {
        is_even((n - 2) as nat)
    }
}

// Evidence that a number is even, mirroring:
//   ev_0 : even 0
//   ev_SS : even n -> even (S (S n))
#[verifier::accept_recursive_types]
pub enum EvenEv {
    Ev0,
    EvSS(Box<EvenEv>),
}

impl EvenEv {
    pub open spec fn n(self) -> nat
        decreases self
    {
        match self {
            EvenEv::Ev0 => 0,
            EvenEv::EvSS(e) => (*e).n() + 2,
        }
    }
}

// 1) soundness: evidence implies computational predicate
pub proof fn ex1_even_sound(e: EvenEv)
    ensures is_even(e.n())
    decreases e
{
    match e {
        EvenEv::Ev0 => {
            assert(is_even(0));
        }
        EvenEv::EvSS(e1) => {
            ex1_even_sound(*e1);
            assert(is_even(e.n()) == is_even((e1.n()) as nat));
        }
    }
}

// 2) completeness: computational predicate implies evidence
pub proof fn ex2_even_complete(n: nat) -> (e: EvenEv)
    requires is_even(n),
    ensures e.n() == n,
    decreases n
{
    if n == 0 {
        EvenEv::Ev0
    } else {
        // For nat, if is_even(n) and n != 0, then n >= 2.
        assert(n != 1);
        assert(n >= 2);
        let n2 = (n - 2) as nat;
        let e2 = ex2_even_complete(n2);
        EvenEv::EvSS(Box::new(e2))
    }
}

// 3) inversion-style: any even evidence for n>0 must be EvSS
pub proof fn ex3_even_inversion(e: EvenEv)
    requires e.n() > 0,
    ensures match e { EvenEv::EvSS(_) => true, _ => false }
{
    match e {
        EvenEv::Ev0 => {
            assert(e.n() == 0);
            assert(false);
        }
        EvenEv::EvSS(_) => {}
    }
}

// ------------------------
// <= (inductive relation)
// ------------------------

// Evidence for `lhs <= rhs`, mirroring:
//   le_n : n <= n
//   le_S : n <= m -> n <= S m
#[verifier::accept_recursive_types]
pub enum LeEv {
    Refl(nat),
    Step(Box<LeEv>),
}

impl LeEv {
    pub open spec fn lhs(self) -> nat
        decreases self
    {
        match self {
            LeEv::Refl(n) => n,
            LeEv::Step(p) => (*p).lhs(),
        }
    }

    pub open spec fn rhs(self) -> nat
        decreases self
    {
        match self {
            LeEv::Refl(n) => n,
            LeEv::Step(p) => (*p).rhs() + 1,
        }
    }
}

// 4) reflexivity
pub proof fn ex4_le_refl(n: nat) -> (p: LeEv)
    ensures p.lhs() == n && p.rhs() == n
{
    LeEv::Refl(n)
}

// 5) one-step: n <= n+1
pub proof fn ex5_le_succ(n: nat) -> (p: LeEv)
    ensures p.lhs() == n && p.rhs() == n + 1
{
    let p = LeEv::Step(Box::new(LeEv::Refl(n)));
    reveal_with_fuel(LeEv::lhs, 3);
    reveal_with_fuel(LeEv::rhs, 3);
    assert(p.lhs() == n);
    assert(p.rhs() == n + 1);
    p
}

// 6) transitivity (compose evidence)
pub proof fn ex6_le_trans(p: LeEv, q: LeEv) -> (r: LeEv)
    requires p.rhs() == q.lhs(),
    ensures r.lhs() == p.lhs() && r.rhs() == q.rhs(),
    decreases q
{
    match q {
        LeEv::Refl(_) => {
            // q.rhs == q.lhs, so p already has rhs == q.rhs.
            p
        }
        LeEv::Step(q1) => {
            let mid = ex6_le_trans(p, *q1);
            LeEv::Step(Box::new(mid))
        }
    }
}

// 7) soundness: evidence implies numeric <=
pub proof fn ex7_le_sound(p: LeEv)
    ensures p.lhs() <= p.rhs()
    decreases p
{
    match p {
        LeEv::Refl(_) => {}
        LeEv::Step(p1) => {
            ex7_le_sound(*p1);
            assert(p.lhs() == p1.lhs());
            assert(p.rhs() == p1.rhs() + 1);
        }
    }
}

// ------------------------
// Palindrome (inductive)
// ------------------------

#[verifier::accept_recursive_types]
pub enum PalEv<A> {
    Empty,
    Single(A),
    Step(A, Box<PalEv<A>>),
}

impl<A> PalEv<A> {
    pub open spec fn seq(self) -> Seq<A>
        decreases self
    {
        match self {
            PalEv::Empty => Seq::empty(),
            PalEv::Single(a) => seq![a],
            PalEv::Step(a, mid) => seq![a].add((*mid).seq()).add(seq![a]),
        }
    }
}

pub proof fn lemma_reverse_index<A>(s: Seq<A>, i: int)
    requires 0 <= i < s.len(),
    ensures s.reverse()[i] == s[s.len() - 1 - i]
{
    reveal_with_fuel(Seq::reverse, 1);
    assert(s.reverse()[i] == s[s.len() - 1 - i]);
}

// 8) palindrome evidence implies sequence equals its reverse
pub proof fn ex8_pal_sound<A>(p: PalEv<A>)
    ensures p.seq() =~= p.seq().reverse()
    decreases p
{
    match p {
        PalEv::Empty => {
            assert(Seq::<A>::empty().reverse() =~= Seq::<A>::empty());
        }
        PalEv::Single(a) => {
            reveal_with_fuel(Seq::reverse, 1);
            assert(seq![a].reverse() =~= seq![a]);
        }
        PalEv::Step(a, mid) => {
            ex8_pal_sound(*mid);
            let s = seq![a].add(mid.seq()).add(seq![a]);
            let rs = s.reverse();

            // Prove s == reverse(s) by extensional equality.
            assert(s.len() == rs.len());
            assert forall|i: int| 0 <= i < s.len() implies s[i] == rs[i] by {
                lemma_reverse_index(s, i);
                assert(rs[i] == s[s.len() - 1 - i]);

                if i == 0 {
                    assert(s[0] == a);
                } else if i == s.len() as int - 1 {
                    assert(s[s.len() as int - 1] == a);
                } else {
                    // Middle region: reduce to mid by symmetry.
                    // s = [a] ++ mid ++ [a]
                    assert(0 < i);
                    assert(i < s.len() as int - 1);

                    let j = i - 1;
                    assert(0 <= j);
                    assert(j < mid.seq().len());

                    // Show s[i] is mid[j]
                    assert(s[i] == mid.seq()[j]);

                    let k = (s.len() as int - 1 - i) - 1;
                    assert(0 <= k);
                    assert(k < mid.seq().len());
                    assert(s[s.len() as int - 1 - i] == mid.seq()[k]);

                    // Use IH: mid == reverse(mid)
                    assert(mid.seq() =~= mid.seq().reverse());
                    assert(mid.seq()[j] == mid.seq().reverse()[j]);
                    lemma_reverse_index(mid.seq(), j);
                    assert(mid.seq().reverse()[j] == mid.seq()[mid.seq().len() - 1 - j]);

                    // Connect: k == len(mid)-1-j
                    assert(k == mid.seq().len() as int - 1 - j);
                    assert(mid.seq()[j] == mid.seq()[k]);
                }
            };
            assert(s =~= rs);
        }
    }
}

pub proof fn lemma_is_even_6()
    ensures is_even(6)
{
    reveal_with_fuel(is_even, 4);
    assert(is_even(6));
}

pub fn main() {
    proof {
        // Even
        let e0 = EvenEv::Ev0;
        ex1_even_sound(e0);
        let e4 = EvenEv::EvSS(Box::new(EvenEv::EvSS(Box::new(EvenEv::Ev0))));
        ex1_even_sound(e4);
        lemma_is_even_6();
        let e_from = ex2_even_complete(6);
        ex1_even_sound(e_from);

        // Le
        let p = ex4_le_refl(5);
        ex7_le_sound(p);
        let p1 = ex5_le_succ(5);
        ex7_le_sound(p1);
        let p2 = ex5_le_succ(6);
        let t = ex6_le_trans(p1, p2);
        ex7_le_sound(t);

        // Palindrome
        let pal: PalEv<nat> = PalEv::Step(1nat, Box::new(PalEv::Step(2nat, Box::new(PalEv::Single(3nat)))));
        ex8_pal_sound(pal);
    }
}

} // verus!
