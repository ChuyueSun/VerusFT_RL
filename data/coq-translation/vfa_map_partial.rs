use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// VFA: Partial Maps (vfa-current/Maps)
// Partial maps (may not have all keys)
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Partial Map Type
// ----------------------------------------------------------------------------

pub struct PartialMap<V> {
    pub map: Map<nat, V>,
}

// ----------------------------------------------------------------------------
// Operations
// ----------------------------------------------------------------------------

// Create empty partial map
pub open spec fn p_empty<V>() -> PartialMap<V> {
    PartialMap {
        map: Map::<nat, V>::empty(),
    }
}

// Check if key is bound
pub open spec fn p_bound<V>(m: PartialMap<V>, k: nat) -> bool {
    m.map.dom().contains(k)
}

// Get value (option-like semantics encoded as tuple)
pub open spec fn p_get<V>(m: PartialMap<V>, k: nat) -> Option<V> {
    if m.map.dom().contains(k) {
        Some(m.map[k])
    } else {
        None
    }
}

// Update value
pub open spec fn p_update<V>(m: PartialMap<V>, k: nat, v: V) -> PartialMap<V> {
    PartialMap {
        map: m.map.insert(k, v),
    }
}

// Remove key
pub open spec fn p_remove<V>(m: PartialMap<V>, k: nat) -> PartialMap<V> {
    PartialMap {
        map: m.map.remove(k),
    }
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Empty map has no bindings
pub proof fn p_bound_empty<V>(k: nat)
    ensures !p_bound::<V>(p_empty(), k)
{
}

// Get from empty returns None
pub proof fn p_get_empty<V>(k: nat)
    ensures p_get::<V>(p_empty(), k).is_none()
{
}

// Update makes key bound
pub proof fn p_bound_update<V>(m: PartialMap<V>, k: nat, v: V)
    ensures p_bound(p_update(m, k, v), k)
{
}

// Get after update (same key)
pub proof fn p_get_update_eq<V>(m: PartialMap<V>, k: nat, v: V)
    ensures p_get(p_update(m, k, v), k) == Some(v)
{
}

// Get after update (different key)
pub proof fn p_get_update_neq<V>(m: PartialMap<V>, k1: nat, k2: nat, v: V)
    requires k1 != k2
    ensures p_get(p_update(m, k1, v), k2) == p_get(m, k2)
{
}

// Remove makes key unbound
pub proof fn p_bound_remove<V>(m: PartialMap<V>, k: nat)
    ensures !p_bound(p_remove(m, k), k)
{
}

// Get after remove (same key)
pub proof fn p_get_remove_eq<V>(m: PartialMap<V>, k: nat)
    ensures p_get(p_remove(m, k), k).is_none()
{
}

// Get after remove (different key)
pub proof fn p_get_remove_neq<V>(m: PartialMap<V>, k1: nat, k2: nat)
    requires k1 != k2
    ensures p_get(p_remove(m, k1), k2) == p_get(m, k2)
{
}

// ----------------------------------------------------------------------------
// Shadow and Permutation Properties
// ----------------------------------------------------------------------------

// Update shadow (last update wins)
pub proof fn p_update_shadow<V>(m: PartialMap<V>, k: nat, v1: V, v2: V)
    ensures p_get(p_update(p_update(m, k, v1), k, v2), k) == Some(v2)
{
}

// Update permutation
pub proof fn p_update_permute<V>(m: PartialMap<V>, k1: nat, v1: V, k2: nat, v2: V, k: nat)
    requires k1 != k2
    ensures p_get(p_update(p_update(m, k1, v1), k2, v2), k) ==
            p_get(p_update(p_update(m, k2, v2), k1, v1), k)
{
    if k == k1 {
        p_get_update_eq(p_update(m, k2, v2), k1, v1);
        p_get_update_neq(p_update(m, k1, v1), k1, k2, v2);
        p_get_update_eq(m, k1, v1);
    } else if k == k2 {
        p_get_update_eq(p_update(m, k1, v1), k2, v2);
        p_get_update_neq(p_update(m, k2, v2), k2, k1, v1);
        p_get_update_eq(m, k2, v2);
    } else {
        p_get_update_neq(p_update(m, k1, v1), k, k2, v2);
        p_get_update_neq(m, k, k1, v1);
        p_get_update_neq(p_update(m, k2, v2), k, k1, v1);
        p_get_update_neq(m, k, k2, v2);
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_empty()
{
    p_bound_empty::<nat>(5);
    assert(!p_bound::<nat>(p_empty(), 5));

    p_get_empty::<nat>(5);
    assert(p_get::<nat>(p_empty(), 5).is_none());
}

pub proof fn example_update_get()
{
    let m: PartialMap<nat> = p_empty();
    let m1 = p_update(m, 5, 50);

    p_get_update_eq(m, 5, 50nat);
    assert(p_get(m1, 5) == Some(50nat));

    p_get_update_neq(m, 5, 3, 50nat);
    assert(p_get(m1, 3).is_none());
}

pub proof fn example_remove()
{
    let m: PartialMap<nat> = p_update(p_empty(), 5, 50);
    let m1 = p_remove(m, 5);

    p_bound_remove(m, 5);
    assert(!p_bound(m1, 5));

    p_get_remove_eq(m, 5);
    assert(p_get(m1, 5).is_none());
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn map_partial_verify()
{
    example_empty();
    example_update_get();
    example_remove();

    // Test shadow property
    let m: PartialMap<nat> = p_empty();
    p_update_shadow(m, 1, 10nat, 20);
}

pub fn main() {
    proof {
        map_partial_verify();
    }
}

} // verus!
