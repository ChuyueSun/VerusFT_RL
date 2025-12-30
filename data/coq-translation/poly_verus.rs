use vstd::prelude::*;
use vstd::seq::group_seq_axioms;

verus! {

broadcast use group_seq_axioms;

// Poly (Software Foundations, LF/Poly) style snippets in Verus.
// We use `Seq<A>` as the polymorphic list type.

pub type List<A> = Seq<A>;

pub open spec fn length<A>(xs: List<A>) -> nat {
    xs.len()
}

pub open spec fn app<A>(xs: List<A>, ys: List<A>) -> List<A> {
    xs.add(ys)
}

pub open spec fn map<A, B>(xs: List<A>, f: spec_fn(A) -> B) -> List<B> {
    Seq::new(xs.len(), |i: int| f(xs[i]))
}

pub open spec fn filter<A>(xs: List<A>, p: spec_fn(A) -> bool) -> List<A>
    decreases xs.len()
{
    if xs.len() == 0 {
        Seq::empty()
    } else {
        let x = xs[0];
        let rest = filter(xs.skip(1), p);
        if p(x) {
            seq![x].add(rest)
        } else {
            rest
        }
    }
}

pub open spec fn fold_right<A, B>(xs: List<A>, init: B, f: spec_fn(A, B) -> B) -> B
    decreases xs.len()
{
    if xs.len() == 0 {
        init
    } else {
        f(xs[0], fold_right(xs.skip(1), init, f))
    }
}

pub open spec fn option_map<A, B>(o: Option<A>, f: spec_fn(A) -> B) -> Option<B> {
    match o {
        Option::None => Option::None,
        Option::Some(x) => Option::Some(f(x)),
    }
}

pub proof fn lemma_decompose_head_tail<A>(xs: List<A>)
    requires xs.len() > 0,
    ensures xs =~= seq![xs[0]].add(xs.skip(1))
{
    let tail = xs.skip(1);
    assert(seq![xs[0]].len() == 1);
    assert(tail.len() + 1 == xs.len());

    assert forall|i: int| 0 <= i < xs.len() implies seq![xs[0]].add(tail)[i] == xs[i] by {
        if i == 0 {
            assert(seq![xs[0]].add(tail)[0] == xs[0]);
        } else {
            assert(0 < i);
            assert(i - 1 >= 0);
            assert(i - 1 < tail.len());
            assert(seq![xs[0]].add(tail)[i] == tail[i - 1]);
            assert(tail[i - 1] == xs[i]);
        }
    };

    assert(seq![xs[0]].add(tail) =~= xs);
}

pub proof fn lemma_reverse_index<A>(s: Seq<A>, i: int)
    requires 0 <= i < s.len(),
    ensures s.reverse()[i] == s[s.len() - 1 - i]
{
    reveal_with_fuel(Seq::reverse, 1);
    assert(s.reverse()[i] == s[s.len() - 1 - i]);
}

// 1) map_id
pub proof fn ex1_map_id<A>(xs: List<A>)
    ensures map(xs, |a: A| a) =~= xs
{
    assert(map(xs, |a: A| a).len() == xs.len());
    assert forall|i: int| 0 <= i < xs.len() implies map(xs, |a: A| a)[i] == xs[i] by {
        assert(map(xs, |a: A| a)[i] == (|a: A| a)(xs[i]));
    };
    assert(map(xs, |a: A| a) =~= xs);
}

// 2) map composition
pub proof fn ex2_map_comp<A, B, C>(xs: List<A>, f: spec_fn(A) -> B, g: spec_fn(B) -> C)
    ensures map(map(xs, f), g) =~= map(xs, |a: A| g(f(a)))
{
    let left = map(map(xs, f), g);
    let right = map(xs, |a: A| g(f(a)));
    assert(left.len() == right.len());
    assert forall|i: int| 0 <= i < xs.len() implies left[i] == right[i] by {
        // left[i] = g(map(xs,f)[i]) = g(f(xs[i]))
        assert(map(xs, f).len() == xs.len());
        assert(left[i] == g(map(xs, f)[i]));
        assert(map(xs, f)[i] == f(xs[i]));
    };
    assert(left =~= right);
}

// 3) length(map xs f) = length xs
pub proof fn ex3_length_map<A, B>(xs: List<A>, f: spec_fn(A) -> B)
    ensures length(map(xs, f)) == length(xs)
{
    assert(map(xs, f).len() == xs.len());
}

// 4) map distributes over append
pub proof fn ex4_map_app<A, B>(xs: List<A>, ys: List<A>, f: spec_fn(A) -> B)
    ensures map(xs.add(ys), f) =~= map(xs, f).add(map(ys, f))
{
    let left = map(xs.add(ys), f);
    let right = map(xs, f).add(map(ys, f));

    let m: int = xs.len() as int;
    let n: int = ys.len() as int;
    let total: int = m + n;

    assert(left.len() == total as nat);
    assert(right.len() == total as nat);

    assert forall|i: int| 0 <= i < total implies left[i] == right[i] by {
        if i < m {
            assert(xs.add(ys)[i] == xs[i]);
            assert(left[i] == f(xs[i]));
            assert(right[i] == map(xs, f)[i]);
            assert(map(xs, f)[i] == f(xs[i]));
        } else {
            assert(i - m >= 0);
            assert(i - m < n);
            assert(xs.add(ys)[i] == ys[i - m]);
            assert(left[i] == f(ys[i - m]));
            assert(right[i] == map(ys, f)[i - m]);
            assert(map(ys, f)[i - m] == f(ys[i - m]));
        }
    };

    assert(left =~= right);
}

// 5) filter with always-true predicate is identity
pub proof fn ex5_filter_true<A>(xs: List<A>)
    ensures filter(xs, |a: A| true) =~= xs
    decreases xs.len()
{
    if xs.len() == 0 {
        assert(filter(xs, |a: A| true) =~= Seq::empty());
        assert(xs =~= Seq::empty());
    } else {
        let tail = xs.skip(1);
        ex5_filter_true(tail);
        lemma_decompose_head_tail(xs);
        assert(filter(xs, |a: A| true) == seq![xs[0]].add(filter(tail, |a: A| true)));
        assert(filter(xs, |a: A| true) =~= seq![xs[0]].add(tail));
    }
}

// 6) filter with always-false predicate yields empty
pub proof fn ex6_filter_false<A>(xs: List<A>)
    ensures filter(xs, |a: A| false) =~= Seq::empty()
    decreases xs.len()
{
    if xs.len() == 0 {
        assert(filter(xs, |a: A| false) =~= Seq::empty());
    } else {
        let tail = xs.skip(1);
        ex6_filter_false(tail);
        assert(filter(xs, |a: A| false) == filter(tail, |a: A| false));
    }
}

// 7) filter does not increase length
pub proof fn ex7_filter_len_le<A>(xs: List<A>, p: spec_fn(A) -> bool)
    ensures filter(xs, p).len() <= xs.len()
    decreases xs.len()
{
    if xs.len() == 0 {
        assert(filter(xs, p).len() == 0);
    } else {
        let tail = xs.skip(1);
        ex7_filter_len_le(tail, p);
        if p(xs[0]) {
            assert(filter(xs, p) == seq![xs[0]].add(filter(tail, p)));
            assert(filter(xs, p).len() == filter(tail, p).len() + 1);
            assert(filter(tail, p).len() + 1 <= tail.len() + 1);
            assert(tail.len() + 1 == xs.len());
        } else {
            assert(filter(xs, p) == filter(tail, p));
            assert(filter(xs, p).len() <= tail.len());
            assert(tail.len() < xs.len());
        }
    }
}

// 8) fold_right unfolds on nonempty lists
pub proof fn ex8_fold_right_unfold<A, B>(xs: List<A>, init: B, f: spec_fn(A, B) -> B)
    requires xs.len() > 0,
    ensures fold_right(xs, init, f) == f(xs[0], fold_right(xs.skip(1), init, f))
{
    reveal_with_fuel(fold_right, 1);
    assert(fold_right(xs, init, f) == f(xs[0], fold_right(xs.skip(1), init, f)));
}

// 9) option_map identity
pub proof fn ex9_option_map_id<A>(o: Option<A>)
    ensures option_map(o, |a: A| a) == o
{
    match o {
        Option::None => {
            assert(option_map(o, |a: A| a) == Option::<A>::None);
        }
        Option::Some(x) => {
            assert(option_map(o, |a: A| a) == Option::Some(x));
        }
    }
}

// 10) option_map composition
pub proof fn ex10_option_map_comp<A, B, C>(o: Option<A>, f: spec_fn(A) -> B, g: spec_fn(B) -> C)
    ensures option_map(option_map(o, f), g) == option_map(o, |a: A| g(f(a)))
{
    match o {
        Option::None => {
            assert(option_map(option_map(o, f), g) == Option::<C>::None);
            assert(option_map(o, |a: A| g(f(a))) == Option::<C>::None);
        }
        Option::Some(x) => {
            assert(option_map(option_map(o, f), g) == Option::Some(g(f(x))));
            assert(option_map(o, |a: A| g(f(a))) == Option::Some(g(f(x))));
        }
    }
}

// 11) map commutes with reverse (a common Poly/list exercise)
pub proof fn ex11_map_reverse_commute<A, B>(xs: List<A>, f: spec_fn(A) -> B)
    ensures map(xs.reverse(), f) =~= map(xs, f).reverse()
{
    let n: int = xs.len() as int;
    reveal_with_fuel(Seq::reverse, 1);
    assert(xs.reverse().len() == xs.len());
    assert(map(xs.reverse(), f).len() == xs.len());
    assert(map(xs, f).reverse().len() == xs.len());

    assert forall|i: int| 0 <= i < n implies map(xs.reverse(), f)[i] == map(xs, f).reverse()[i] by {
        // LHS indexing on map and reverse
        lemma_reverse_index(xs, i);
        assert(xs.reverse()[i] == xs[n - 1 - i]);
        assert(map(xs.reverse(), f)[i] == f(xs.reverse()[i]));
        assert(map(xs.reverse(), f)[i] == f(xs[n - 1 - i]));

        // RHS: reverse(map(xs,f))[i] = map(xs,f)[n-1-i] = f(xs[n-1-i])
        lemma_reverse_index(map(xs, f), i);
        assert(map(xs, f).reverse()[i] == map(xs, f)[n - 1 - i]);
        assert(map(xs, f)[n - 1 - i] == f(xs[n - 1 - i]));
    };

    assert(map(xs.reverse(), f) =~= map(xs, f).reverse());
}

pub fn main() {
    // Concrete instantiations (like SF examples).
    proof {
        let xs: Seq<nat> = seq![1nat, 2, 3, 4];
        let ys: Seq<nat> = seq![10nat, 20];

        ex1_map_id(xs);
        ex3_length_map(xs, |a: nat| a + 1);
        ex4_map_app(xs, ys, |a: nat| a + 1);
        ex5_filter_true(xs);
        ex6_filter_false(xs);
        ex7_filter_len_le(xs, |a: nat| a % 2 == 0);

        ex8_fold_right_unfold(xs, 0nat, |a: nat, acc: nat| a + acc);

        ex9_option_map_id::<nat>(Option::Some(5nat));
        ex10_option_map_comp::<nat, nat, nat>(Option::Some(5nat), |a: nat| a + 1, |b: nat| b * 2);

        ex11_map_reverse_commute(xs, |a: nat| a + 1);
    }
}

} // verus!
