use vstd::prelude::*;
use vstd::calc_macro::*;
use vstd::seq_lib::*;
use vstd::seq::group_seq_axioms;

verus! {

broadcast use group_seq_axioms;

// Tactics (Software Foundations, LF/Tactics) style snippets in Verus.
// Goal: demonstrate common proof *moves* (case split, rewriting, calc chains,
// induction, extensionality) in a Verus-friendly way.

pub open spec fn negb(b: bool) -> bool { !b }

pub open spec fn andb(b1: bool, b2: bool) -> bool { b1 && b2 }

pub open spec fn min(x: int, y: int) -> int { if x <= y { x } else { y } }

pub open spec fn add(n: nat, m: nat) -> nat
    decreases n
{
    if n == 0 { m } else { add((n - 1) as nat, m) + 1 }
}

// 1) destruct / case split
pub proof fn ex1_negb_involutive(b: bool)
    ensures negb(negb(b)) == b
{
    match b {
        true => {
            assert(negb(true) == false);
            assert(negb(negb(true)) == true);
        }
        false => {
            assert(negb(false) == true);
            assert(negb(negb(false)) == false);
        }
    }
}

// 2) simplification of an if-expression
pub proof fn ex2_if_same<T>(b: bool, x: T)
    ensures (if b { x } else { x }) == x
{
    if b {
        assert((if b { x } else { x }) == x);
    } else {
        assert((if b { x } else { x }) == x);
    }
}

// 3) induction over nat (like `induction n`)
pub proof fn ex3_add_n_0(n: nat)
    ensures add(n, 0) == n
    decreases n
{
    if n == 0 {
        assert(add(0, 0) == 0);
    } else {
        let n1 = (n - 1) as nat;
        ex3_add_n_0(n1);
        assert(add(n, 0) == add(n1, 0) + 1);
        assert(n == n1 + 1);
    }
}

// 4) induction + rewriting (assoc)
pub proof fn ex4_add_assoc(a: nat, b: nat, c: nat)
    ensures add(add(a, b), c) == add(a, add(b, c))
    decreases a
{
    if a == 0 {
        assert(add(add(0, b), c) == add(b, c));
        assert(add(0, add(b, c)) == add(b, c));
    } else {
        let a1 = (a - 1) as nat;
        ex4_add_assoc(a1, b, c);

        assert(add(a, b) == add(a1, b) + 1);
        assert(add(add(a, b), c) == add(add(a1, b) + 1, c));
        assert(add(a, add(b, c)) == add(a1, add(b, c)) + 1);

        // A tiny rewrite: add(x+1, c) unfolds to add(x,c)+1
        assert(add(add(a1, b) + 1, c) == add(add(a1, b), c) + 1);

        // Close with IH
        assert(add(add(a1, b), c) == add(a1, add(b, c)));
    }
}

// 5) calc chain (like `rewrite` / `transitivity` / `calc`)
pub proof fn ex5_calc_assoc(a: nat, b: nat, c: nat)
    ensures add(add(a, b), c) == add(a, add(b, c))
{
    calc! {
        (==)
        add(add(a, b), c);
        { ex4_add_assoc(a, b, c); }
        add(a, add(b, c));
    }
}

// 6) forall-intro style reasoning
pub proof fn ex6_min_le_both()
    ensures forall|i: int, j: int| min(i, j) <= i && min(i, j) <= j
{
    assert forall|i: int, j: int| min(i, j) <= i && min(i, j) <= j by {
        if i <= j {
            assert(min(i, j) == i);
        } else {
            assert(min(i, j) == j);
        }
    };
}

// 7) extensionality (like `extensionality` + `simpl`)
pub proof fn ex7_subrange_concat<A>(s: Seq<A>, i: int)
    requires 0 <= i <= s.len(),
    ensures s =~= s.subrange(0, i).add(s.subrange(i, s.len() as int))
{
    let t1 = s.subrange(0, i);
    let t2 = s.subrange(i, s.len() as int);
    let t = t1.add(t2);

    assert_seqs_equal!(s == t);
    assert(s =~= t);
}

// 8) `destruct` on Option + `simpl`
pub open spec fn option_map<A, B>(o: Option<A>, f: spec_fn(A) -> B) -> Option<B> {
    match o {
        Option::None => Option::None,
        Option::Some(x) => Option::Some(f(x)),
    }
}

pub proof fn ex8_option_map_none<A, B>(f: spec_fn(A) -> B)
    ensures option_map(Option::<A>::None, f) == Option::<B>::None
{
    assert(option_map(Option::<A>::None, f) == Option::<B>::None);
}

// 9) exists-witness (like `exists` + `reflexivity`)
pub proof fn ex9_exists_nat()
    ensures exists|k: nat| #[trigger] add(k, 0) == k
{
    assert(exists|k: nat| #[trigger] add(k, 0) == k) by {
        let w: nat = 0;
        ex3_add_n_0(w);
        assert(add(w, 0) == w);
    };
}

// 10) contradiction (like `discriminate` / `contradiction`)
pub proof fn ex10_nat_nonzero_implies_not_zero(n: nat)
    requires n > 0,
    ensures n != 0
{
    if n == 0 {
        // Impossible branch given precondition.
        assert(false);
    }
}

pub fn main() {
    proof {
        ex1_negb_involutive(true);
        ex1_negb_involutive(false);

        ex2_if_same(true, 5nat);
        ex2_if_same(false, 5nat);

        ex3_add_n_0(0);
        ex3_add_n_0(10);

        ex4_add_assoc(3, 4, 5);
        ex5_calc_assoc(6, 7, 8);

        ex6_min_le_both();

        ex7_subrange_concat(seq![1nat, 2, 3, 4, 5], 2);

        ex8_option_map_none::<nat, nat>(|a: nat| a + 1);
        ex9_exists_nat();
        ex10_nat_nonzero_implies_not_zero(5);
    }
}

} // verus!
