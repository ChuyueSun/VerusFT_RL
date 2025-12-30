use vstd::prelude::*;
use vstd::seq_lib::*;
use vstd::seq::group_seq_axioms;

verus! {

broadcast use group_seq_axioms;

// Lists (Software Foundations, LF/Lists) style snippets in Verus.
// We model Coq's `list nat` using Verus's `Seq<nat>`.

pub type NatList = Seq<nat>;

pub open spec fn length(xs: NatList) -> nat {
    xs.len()
}

pub open spec fn app(xs: NatList, ys: NatList) -> NatList {
    xs.add(ys)
}

pub open spec fn snoc(xs: NatList, v: nat) -> NatList {
    xs.push(v)
}

pub open spec fn repeat(v: nat, count: nat) -> NatList {
    Seq::new(count, |i: int| v)
}

pub open spec fn hd(default: nat, xs: NatList) -> nat {
    if xs.len() == 0 {
        default
    } else {
        xs[0]
    }
}

pub open spec fn tl(xs: NatList) -> NatList {
    if xs.len() == 0 {
        xs
    } else {
        xs.skip(1)
    }
}

pub open spec fn nonzeros(xs: NatList) -> NatList
    decreases xs.len()
{
    if xs.len() == 0 {
        Seq::empty()
    } else {
        let x = xs[0];
        let rest = nonzeros(xs.skip(1));
        if x == 0 {
            rest
        } else {
            seq![x].add(rest)
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

// 1) length(repeat v n) = n
pub proof fn ex1_length_repeat(v: nat, n: nat)
    ensures length(repeat(v, n)) == n
{
    assert(repeat(v, n).len() == n);
}

// 2) app_nil_r
pub proof fn ex2_app_nil_r(xs: NatList)
    ensures app(xs, Seq::empty()) =~= xs
{
    assert(xs.add(Seq::empty()) =~= xs);
}

// 3) app_assoc
pub proof fn ex3_app_assoc(xs: NatList, ys: NatList, zs: NatList)
    ensures app(app(xs, ys), zs) =~= app(xs, app(ys, zs))
{
    lemma_concat_associative(xs, ys, zs);
    assert(xs.add(ys).add(zs) =~= xs.add(ys.add(zs)));
}

// 4) length(app xs ys) = length xs + length ys
pub proof fn ex4_length_app(xs: NatList, ys: NatList)
    ensures length(app(xs, ys)) == length(xs) + length(ys)
{
    assert(xs.add(ys).len() == xs.len() + ys.len());
}

// 5) snoc xs v = xs ++ [v]
pub proof fn ex5_snoc_is_app_singleton(xs: NatList, v: nat)
    ensures snoc(xs, v) =~= app(xs, seq![v])
{
    assert(xs.push(v) =~= xs.add(seq![v]));
}

// 6) tl drops one element when nonempty
pub proof fn ex6_tl_len(xs: NatList)
    requires xs.len() > 0,
    ensures tl(xs).len() + 1 == xs.len()
{
    assert(tl(xs) == xs.skip(1));
    assert(xs.skip(1).len() + 1 == xs.len());
}

// 7) reverse involutive
pub proof fn ex7_rev_involutive(xs: NatList)
    ensures xs.reverse().reverse() =~= xs
{
    // Prove by extensional equality (same length, same indexing).
    assert(xs.reverse().len() == xs.len());
    assert(xs.reverse().reverse().len() == xs.len());

    assert forall|i: int| 0 <= i < xs.len() implies xs.reverse().reverse()[i] == xs[i] by {
        lemma_reverse_index(xs.reverse(), i);
        lemma_reverse_index(xs, xs.len() - 1 - i);

        // Expand the two reverses.
        assert(xs.reverse().reverse()[i] == xs.reverse()[xs.len() - 1 - i]);
        assert(xs.reverse()[xs.len() - 1 - i] == xs[xs.len() - 1 - (xs.len() - 1 - i)]);
        assert(xs.len() - 1 - (xs.len() - 1 - i) == i);
    };

    assert(xs.reverse().reverse() =~= xs);
}

// 8) reverse(snoc xs v) = [v] ++ reverse(xs)
pub proof fn ex8_rev_snoc(xs: NatList, v: nat)
    ensures xs.push(v).reverse() =~= seq![v].add(xs.reverse())
{
    // One convenient route: prove the stronger concat lemma (next) and instantiate.
    ex9_rev_app_distr(xs, seq![v]);

    // reverse(xs ++ [v]) = reverse([v]) ++ reverse(xs)
    assert(xs.add(seq![v]).reverse() =~= seq![v].reverse().add(xs.reverse()));
    reveal_with_fuel(Seq::reverse, 1);
    assert(seq![v].reverse() =~= seq![v]);
}

// 9) reverse distributes over append
pub proof fn ex9_rev_app_distr(xs: NatList, ys: NatList)
    ensures xs.add(ys).reverse() =~= ys.reverse().add(xs.reverse())
{
    let m: int = xs.len() as int;
    let n: int = ys.len() as int;
    let total: int = m + n;

    assert(xs.add(ys).len() == (total as nat));
    assert(ys.reverse().add(xs.reverse()).len() == (total as nat));

    assert forall|i: int| 0 <= i < total implies xs.add(ys).reverse()[i] == ys.reverse().add(xs.reverse())[i] by {
        // LHS: reverse(xs ++ ys)[i] = (xs ++ ys)[total-1-i]
        lemma_reverse_index(xs.add(ys), i);
        assert(xs.add(ys).reverse()[i] == xs.add(ys)[total - 1 - i]);

        if i < n {
            // Then total-1-i >= m, so index is in the ys part.
            assert(total - 1 - i >= m);
            assert(xs.add(ys)[total - 1 - i] == ys[(total - 1 - i) - m]);
            assert((total - 1 - i) - m == n - 1 - i);

            // RHS index i is in ys.reverse.
            lemma_reverse_index(ys, i);
            assert(ys.reverse()[i] == ys[n - 1 - i]);
            assert(ys.reverse().add(xs.reverse())[i] == ys.reverse()[i]);
        } else {
            // Then i-n is a valid index into xs.reverse.
            assert(i - n >= 0);
            assert(i - n < m);

            // total-1-i < m, so index is in the xs part.
            assert(total - 1 - i < m);
            assert(xs.add(ys)[total - 1 - i] == xs[total - 1 - i]);
            assert(total - 1 - i == m - 1 - (i - n));

            // RHS index i is in the xs.reverse part.
            lemma_reverse_index(xs, i - n);
            assert(ys.reverse().add(xs.reverse())[i] == xs.reverse()[i - n]);
            assert(xs.reverse()[i - n] == xs[m - 1 - (i - n)]);
        }
    };

    assert(xs.add(ys).reverse() =~= ys.reverse().add(xs.reverse()));
}

// 10) hd of a cons-like list
pub proof fn ex10_hd_cons(d: nat, x: nat, xs: NatList)
    ensures hd(d, seq![x].add(xs)) == x
{
    assert(seq![x].add(xs).len() > 0);
    assert(hd(d, seq![x].add(xs)) == (seq![x].add(xs))[0]);
    assert((seq![x].add(xs))[0] == x);
}

// 11) nonzeros never increases length
pub proof fn ex11_nonzeros_len_le(xs: NatList)
    ensures nonzeros(xs).len() <= xs.len()
    decreases xs.len()
{
    if xs.len() == 0 {
        assert(nonzeros(xs).len() == 0);
    } else {
        let tail = xs.skip(1);
        ex11_nonzeros_len_le(tail);

        let x = xs[0];
        if x == 0 {
            assert(nonzeros(xs) == nonzeros(tail));
            assert(nonzeros(xs).len() <= tail.len());
            assert(tail.len() < xs.len());
        } else {
            assert(nonzeros(xs) == seq![x].add(nonzeros(tail)));
            assert(nonzeros(xs).len() == nonzeros(tail).len() + 1);
            assert(nonzeros(tail).len() <= tail.len());
            assert(nonzeros(xs).len() <= tail.len() + 1);
            assert(tail.len() + 1 == xs.len());
        }
    }
}

pub fn main() {
    proof {
        ex1_length_repeat(7, 10);
        ex2_app_nil_r(seq![1nat, 2, 3]);
        ex3_app_assoc(seq![1nat], seq![2nat, 3], seq![4nat]);
        ex4_length_app(seq![1nat, 2], seq![3nat, 4, 5]);
        ex5_snoc_is_app_singleton(seq![1nat, 2], 9);
        ex6_tl_len(seq![8nat, 9, 10]);
        ex7_rev_involutive(seq![1nat, 2, 3, 4]);
        ex8_rev_snoc(seq![1nat, 2, 3], 7);
        ex9_rev_app_distr(seq![1nat, 2], seq![3nat, 4, 5]);
        ex10_hd_cons(0, 42, seq![1nat, 2, 3]);
        ex11_nonzeros_len_le(seq![0nat, 1, 0, 2, 3, 0]);
    }
}

} // verus!
