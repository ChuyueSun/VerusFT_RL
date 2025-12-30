use vstd::prelude::*;
use vstd::seq_lib::*;
use vstd::seq::group_seq_axioms;
use vstd::map::*;
use vstd::set::*;
use vstd::calc;

verus! {

// Auto (Software Foundations, LF/Auto) style snippets in Verus.
//
// Theme: make proofs shorter by (1) structuring them well, (2) reusing lemmas,
// (3) enabling helpful axioms, and (4) using proof combinators/macros.

broadcast use { group_seq_axioms, group_map_axioms, group_set_axioms };

// ----------------------------
// Small arithmetic facts (mostly automated)
// ----------------------------

pub proof fn lemma_nat_add_zero_right(n: nat)
    ensures n + 0 == n
{
}

pub proof fn lemma_nat_add_assoc(a: nat, b: nat, c: nat)
    ensures (a + b) + c == a + (b + c)
{
}

pub proof fn lemma_nat_add_comm(a: nat, b: nat)
    ensures a + b == b + a
{
}

pub proof fn ex_calc_chain(a: nat, b: nat, c: nat)
    ensures a + (b + c) == c + (b + a)
{
    calc! {
        (==)
        a + (b + c);
        { lemma_nat_add_assoc(a, b, c); }
        (a + b) + c;
        { lemma_nat_add_comm(a + b, c); }
        c + (a + b);
        {
            lemma_nat_add_comm(a, b);
            assert(c + (a + b) == c + (b + a));
        }
        c + (b + a);
    }
}

// ----------------------------
// Sequences: extensional proofs via a helper macro
// ----------------------------

pub type ListN = Seq<nat>;

pub proof fn ex_seq_append_nil_right(xs: ListN)
    ensures xs.add(seq![]) =~= xs
{
    assert_seqs_equal!(xs.add(seq![]), xs);
}

pub proof fn ex_seq_append_nil_left(xs: ListN)
    ensures seq![].add(xs) =~= xs
{
    assert_seqs_equal!(seq![].add(xs), xs);
}

// ----------------------------
// Maps: using broadcast axioms to keep proofs short
// ----------------------------

pub type Key = nat;
pub type M = Map<Key, int>;

pub open spec fn m_empty() -> M { Map::<Key, int>::empty() }

pub proof fn lemma_insert_same(m: M, k: Key, v: int)
    ensures m.insert(k, v)[k] == v
{
    axiom_map_insert_same(m, k, v);
}

pub proof fn lemma_insert_different(m: M, k1: Key, k2: Key, v2: int)
    requires k1 != k2
    ensures m.insert(k2, v2)[k1] == m[k1]
{
    axiom_map_insert_different(m, k1, k2, v2);
}

pub proof fn ex_map_update_shadow_value(m: M, k: Key, v1: int, v2: int)
    ensures (m.insert(k, v1).insert(k, v2))[k] == v2
{
    lemma_insert_same(m.insert(k, v1), k, v2);
}

pub fn main() {
    proof {
        ex_calc_chain(1, 2, 3);

        let xs: ListN = seq![1, 2, 3];
        ex_seq_append_nil_right(xs);
        ex_seq_append_nil_left(xs);

        let m = m_empty().insert(0, 10);
        lemma_insert_same(m, 5, 99);
        lemma_insert_different(m, 0, 5, 99);
        ex_map_update_shadow_value(m, 0, 1, 2);
    }
}

} // verus!
