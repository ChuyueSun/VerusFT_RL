use vstd::prelude::*;

verus! {

// Logic (Software Foundations, LF/Logic) style snippets in Verus.
// We model propositions using `bool`-valued spec expressions and quantifiers.

pub open spec fn implies(p: bool, q: bool) -> bool { !p || q }

pub open spec fn iff(p: bool, q: bool) -> bool { implies(p, q) && implies(q, p) }

// 1) and commutativity
pub proof fn ex1_and_comm(p: bool, q: bool)
    ensures (p && q) == (q && p)
{
    match (p, q) {
        (true, true) => {}
        (true, false) => {}
        (false, true) => {}
        (false, false) => {}
    }
}

// 2) or associativity
pub proof fn ex2_or_assoc(p: bool, q: bool, r: bool)
    ensures p || (q || r) == (p || q) || r
{
    match (p, q, r) {
        (false, false, false) => {}
        (false, false, true) => {}
        (false, true, false) => {}
        (false, true, true) => {}
        (true, false, false) => {}
        (true, false, true) => {}
        (true, true, false) => {}
        (true, true, true) => {}
    }
}

// 3) implication transitivity
pub proof fn ex3_imp_trans(p: bool, q: bool, r: bool)
    ensures implies(p, q) && implies(q, r) ==> implies(p, r)
{
    if implies(p, q) && implies(q, r) {
        if p {
            // From p and (p -> q), derive q.
            assert(q);
            // From q and (q -> r), derive r.
            assert(r);
        }
        assert(implies(p, r));
    }
}

// 4) contrapositive
pub proof fn ex4_contrapositive(p: bool, q: bool)
    ensures implies(p, q) ==> implies(!q, !p)
{
    if implies(p, q) {
        if !q {
            if p {
                // p and (p->q) gives q, contradicting !q
                assert(q);
                assert(false);
            }
            assert(!p);
        }
        assert(implies(!q, !p));
    }
}

// 5) De Morgan laws
pub proof fn ex5_demorgan(p: bool, q: bool)
    ensures (!(p || q) == (!p && !q))
        && (!(p && q) == (!p || !q))
{
    match (p, q) {
        (true, true) => {}
        (true, false) => {}
        (false, true) => {}
        (false, false) => {}
    }
}

// 6) excluded middle (classical logic)
pub proof fn ex6_excluded_middle(b: bool)
    ensures b || !b
{
    match b {
        true => {}
        false => {}
    }
}

// 7) double negation for bool
pub proof fn ex7_double_neg(b: bool)
    ensures (!!b) == b
{
    match b {
        true => {}
        false => {}
    }
}

// 8) forall distributes over conjunction
pub proof fn ex8_forall_and_distrib(p: spec_fn(nat) -> bool, q: spec_fn(nat) -> bool)
    ensures (forall|x: nat| #![trigger p(x)] #![trigger q(x)] p(x) && q(x)) <==>
        ((forall|x: nat| #![trigger p(x)] p(x)) && (forall|x: nat| #![trigger q(x)] q(x)))
{
    let lhs = forall|x: nat| #![trigger p(x)] #![trigger q(x)] p(x) && q(x);
    let rhs = (forall|x: nat| #![trigger p(x)] p(x)) && (forall|x: nat| #![trigger q(x)] q(x));

    // lhs ==> rhs
    if lhs {
        assert(rhs) by {
            assert(forall|x: nat| #![trigger p(x)] p(x)) by {
                assert forall|x: nat| #![trigger p(x)] p(x) by {
                    assert(p(x) && q(x));
                };
            };
            assert(forall|x: nat| #![trigger q(x)] q(x)) by {
                assert forall|x: nat| #![trigger q(x)] q(x) by {
                    assert(p(x) && q(x));
                };
            };
        };
    }

    // rhs ==> lhs
    if rhs {
        assert(forall|x: nat| #![trigger p(x)] p(x));
        assert(forall|x: nat| #![trigger q(x)] q(x));
        assert(lhs) by {
            assert forall|x: nat| #![trigger p(x)] #![trigger q(x)] p(x) && q(x) by {
                assert(p(x));
                assert(q(x));
            };
        };
    }

    assert(lhs <==> rhs);
}

// 9) not exists <-> forall not
pub proof fn ex9_not_exists_forall_not(p: spec_fn(nat) -> bool)
    ensures (!(exists|x: nat| #![trigger p(x)] p(x))) <==> (forall|x: nat| #![trigger p(x)] !p(x))
{
    let lhs = !(exists|x: nat| #![trigger p(x)] p(x));
    let rhs = forall|x: nat| #![trigger p(x)] !p(x);

    // lhs ==> rhs
    if lhs {
        assert(rhs) by {
            assert forall|x: nat| #![trigger p(x)] !p(x) by {
                if p(x) {
                    assert(exists|y: nat| #![trigger p(y)] p(y)) by {
                        assert(p(x));
                    };
                    assert(false);
                }
            };
        };
    }

    // rhs ==> lhs
    if rhs {
        if exists|x: nat| #![trigger p(x)] p(x) {
            let w: nat = choose|x: nat| #![trigger p(x)] p(x);
            assert(p(w));
            assert(!p(w));
            assert(false);
        }
        assert(lhs);
    }

    assert(lhs <==> rhs);
}

// 10) exists distributes over or (one direction)
pub proof fn ex10_exists_or(p: spec_fn(nat) -> bool, q: spec_fn(nat) -> bool)
    ensures (exists|x: nat| #![trigger p(x)] #![trigger q(x)] (p(x) || q(x))) ==>
        ((exists|x: nat| #![trigger p(x)] p(x)) || (exists|x: nat| #![trigger q(x)] q(x)))
{
    if exists|x: nat| #![trigger p(x)] #![trigger q(x)] (p(x) || q(x)) {
        let w: nat = choose|x: nat| #![trigger p(x)] #![trigger q(x)] (p(x) || q(x));
        assert(p(w) || q(w));
        if p(w) {
            assert(exists|x: nat| #![trigger p(x)] p(x)) by { assert(p(w)); };
        } else {
            assert(q(w));
            assert(exists|x: nat| #![trigger q(x)] q(x)) by { assert(q(w)); };
        }
    }
}

// 11) universal instantiation: forall x. p(x) -> p(k)
pub proof fn ex11_forall_elim(p: spec_fn(nat) -> bool, k: nat)
    ensures (forall|x: nat| #[trigger] p(x)) ==> p(k)
{
    if forall|x: nat| #[trigger] p(x) {
        assert(p(k));
    }
}

pub fn main() {
    proof {
        ex1_and_comm(true, false);
        ex2_or_assoc(true, false, true);
        ex3_imp_trans(true, true, false);
        ex4_contrapositive(true, false);
        ex5_demorgan(true, false);
        ex6_excluded_middle(true);
        ex6_excluded_middle(false);
        ex7_double_neg(true);
        ex7_double_neg(false);

        // Quantifier examples with concrete predicates.
        let p = |x: nat| x == x;
        let q = |x: nat| x + 0 == x;
        ex8_forall_and_distrib(p, q);
        ex9_not_exists_forall_not(|x: nat| x < 0);
        ex10_exists_or(|x: nat| x == 0, |x: nat| x == 1);
        ex11_forall_elim(|x: nat| x == x, 5);
    }
}

} // verus!
