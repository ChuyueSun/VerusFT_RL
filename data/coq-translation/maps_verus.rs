use vstd::prelude::*;
use vstd::map::*;
use vstd::set::*;

verus! {

// Maps (Software Foundations, LF/Maps) style snippets in Verus.
//
// NOTE: SF models total maps as functions (key -> value) with an `update` operation.
// In this workspace, returning lambdas directly from spec functions triggers a Verus
// limitation (“variable shadowing not yet supported”).
//
// Workaround: represent the map as a finite `Map<Key, int>` of overrides PLUS a default.
// `apply` consults overrides when present, otherwise returns the default.

broadcast use { group_map_axioms, group_set_axioms };

pub type Key = nat;

// Total map = finite overrides + default value
pub type TotalMap = Map<Key, int>;

pub open spec fn t_empty() -> TotalMap {
    Map::<Key, int>::empty()
}

pub open spec fn t_update(m: TotalMap, k: Key, v: int) -> TotalMap {
    m.insert(k, v)
}

pub open spec fn t_apply(m: TotalMap, default: int, k: Key) -> int {
    if m.dom().contains(k) { m[k] } else { default }
}

// 1) update_eq
pub proof fn ex1_update_eq(m: TotalMap, default: int, k: Key, v: int)
    ensures t_apply(t_update(m, k, v), default, k) == v
{
    axiom_map_insert_domain(m, k, v);
    assert(t_update(m, k, v).dom().contains(k));
    axiom_map_insert_same(m, k, v);
    assert(t_apply(t_update(m, k, v), default, k) == t_update(m, k, v)[k]);
}

// 2) update_neq
pub proof fn ex2_update_neq(m: TotalMap, default: int, k1: Key, k2: Key, v: int)
    requires k2 != k1,
    ensures t_apply(t_update(m, k1, v), default, k2) == t_apply(m, default, k2)
{
    axiom_map_insert_domain(m, k1, v);
    if m.dom().contains(k2) {
        // k2 was already mapped; insert at k1 (k1!=k2) preserves value at k2
        assert(t_update(m, k1, v).dom().contains(k2));
        axiom_map_insert_different(m, k2, k1, v);
        assert(t_update(m, k1, v)[k2] == m[k2]);
        assert(t_apply(t_update(m, k1, v), default, k2) == t_update(m, k1, v)[k2]);
        assert(t_apply(m, default, k2) == m[k2]);
    } else {
        // k2 unmapped before; inserting at different key doesn't add k2
        assert(!t_update(m, k1, v).dom().contains(k2));
        assert(t_apply(t_update(m, k1, v), default, k2) == default);
        assert(t_apply(m, default, k2) == default);
    }
}

// 3) update_shadow
pub proof fn ex3_update_shadow(m: TotalMap, default: int, k: Key, v1: int, v2: int)
    ensures forall|x: Key| t_apply(t_update(t_update(m, k, v1), k, v2), default, x)
        == t_apply(t_update(m, k, v2), default, x)
{
    assert forall|x: Key| t_apply(t_update(t_update(m, k, v1), k, v2), default, x)
        == t_apply(t_update(m, k, v2), default, x)
    by {
        if x == k {
            ex1_update_eq(t_update(m, k, v1), default, k, v2);
            ex1_update_eq(m, default, k, v2);
        } else {
            ex2_update_neq(t_update(m, k, v1), default, k, x, v2);
            ex2_update_neq(m, default, k, x, v2);
        }
    };
}

// 4) update_same (pointwise extensionality w.r.t. apply)
pub proof fn ex4_update_same(m: TotalMap, default: int, k: Key)
    ensures forall|x: Key| t_apply(t_update(m, k, t_apply(m, default, k)), default, x) == t_apply(m, default, x)
{
    let v = t_apply(m, default, k);
    assert forall|x: Key| t_apply(t_update(m, k, v), default, x) == t_apply(m, default, x) by {
        if x == k {
            ex1_update_eq(m, default, k, v);
        } else {
            ex2_update_neq(m, default, k, x, v);
        }
    };
}

// 5) update_comm (when keys differ)
pub proof fn ex5_update_comm(m: TotalMap, default: int, k1: Key, v1: int, k2: Key, v2: int)
    requires k1 != k2,
    ensures forall|x: Key| t_apply(t_update(t_update(m, k1, v1), k2, v2), default, x)
        == t_apply(t_update(t_update(m, k2, v2), k1, v1), default, x)
{
    assert forall|x: Key| t_apply(t_update(t_update(m, k1, v1), k2, v2), default, x)
        == t_apply(t_update(t_update(m, k2, v2), k1, v1), default, x)
    by {
        if x == k1 {
            // Left: update at k2 doesn't affect k1
            ex2_update_neq(t_update(m, k1, v1), default, k2, k1, v2);
            ex1_update_eq(t_update(m, k2, v2), default, k1, v1);
        } else if x == k2 {
            ex1_update_eq(t_update(m, k1, v1), default, k2, v2);
            ex2_update_neq(t_update(m, k2, v2), default, k1, k2, v1);
        } else {
            ex2_update_neq(t_update(m, k1, v1), default, k2, x, v2);
            ex2_update_neq(t_update(m, k2, v2), default, k1, x, v1);
        }
    };
}

// ------------------------
// Partial maps (Option-valued)
// ------------------------

pub type PartialMap = Map<Key, int>;

pub open spec fn p_empty() -> PartialMap {
    Map::<Key, int>::empty()
}

pub open spec fn p_update(m: PartialMap, k: Key, v: int) -> PartialMap {
    m.insert(k, v)
}

pub open spec fn p_apply(m: PartialMap, k: Key) -> Option<int> {
    if m.dom().contains(k) { Option::Some(m[k]) } else { Option::<int>::None }
}

// 6) empty is None everywhere
pub proof fn ex6_p_empty_none(k: Key)
    ensures p_apply(p_empty(), k) == Option::<int>::None
{
    // dom(empty) is empty
    assert(Map::<Key, int>::empty().dom() == Set::<Key>::empty());
    assert(!p_empty().dom().contains(k));
}

// 7) p_update_eq
pub proof fn ex7_p_update_eq(m: PartialMap, k: Key, v: int)
    ensures p_apply(p_update(m, k, v), k) == Option::Some(v)
{
    axiom_map_insert_domain(m, k, v);
    assert(p_update(m, k, v).dom().contains(k));
    axiom_map_insert_same(m, k, v);
    assert(p_update(m, k, v)[k] == v);
}

// 8) p_update_neq
pub proof fn ex8_p_update_neq(m: PartialMap, k1: Key, k2: Key, v: int)
    requires k2 != k1,
    ensures p_apply(p_update(m, k1, v), k2) == p_apply(m, k2)
{
    axiom_map_insert_domain(m, k1, v);
    if m.dom().contains(k2) {
        assert(p_update(m, k1, v).dom().contains(k2));
        axiom_map_insert_different(m, k2, k1, v);
    } else {
        assert(!p_update(m, k1, v).dom().contains(k2));
    }
}

// 9) shadow for partial maps
pub proof fn ex9_p_update_shadow(m: PartialMap, k: Key, v1: int, v2: int)
    ensures forall|x: Key| p_apply(p_update(p_update(m, k, v1), k, v2), x)
        == p_apply(p_update(m, k, v2), x)
{
    assert forall|x: Key| p_apply(p_update(p_update(m, k, v1), k, v2), x)
        == p_apply(p_update(m, k, v2), x)
    by {
        if x == k {
            ex7_p_update_eq(p_update(m, k, v1), k, v2);
            ex7_p_update_eq(m, k, v2);
        } else {
            ex8_p_update_neq(p_update(m, k, v1), k, x, v2);
            ex8_p_update_neq(m, k, x, v2);
        }
    };
}

// 10) a small "lookup after updates" example, like SF's Examples
pub proof fn ex10_example_chain()
{
    let default: int = 0;
    let m0: TotalMap = t_empty();
    let m1 = t_update(m0, 1, 42);
    let m2 = t_update(m1, 2, -7);

    assert(t_apply(m2, default, 0) == 0);
    assert(t_apply(m2, default, 1) == 42);
    assert(t_apply(m2, default, 2) == -7);

    let pm0: PartialMap = p_empty();
    let pm1 = p_update(pm0, 5, 99);
    assert(p_apply(pm1, 5) == Option::Some(99int));
    assert(p_apply(pm1, 6) == Option::<int>::None);
}

pub fn main() {
    proof {
        let default: int = 0;
        let m: TotalMap = t_empty();
        ex1_update_eq(m, default, 3, 10);
        ex2_update_neq(m, default, 3, 4, 10);
        ex3_update_shadow(m, default, 3, 10, 20);
        ex4_update_same(m, default, 7);
        ex5_update_comm(m, default, 1, 11, 2, 22);

        let pm: PartialMap = p_empty();
        ex6_p_empty_none(0);
        ex7_p_update_eq(pm, 1, 5);
        ex8_p_update_neq(pm, 1, 2, 5);
        ex9_p_update_shadow(pm, 9, 1, 2);

        ex10_example_chain();
    }
}

} // verus!
