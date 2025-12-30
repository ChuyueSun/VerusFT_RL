use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// VFA: Total Maps (vfa-current/Maps)
// Total maps with default values
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Total Map Type
// ----------------------------------------------------------------------------

// A total map returns a default value for missing keys
pub struct TotalMap<V> {
    pub default: V,
    pub map: Map<nat, V>,
}

// ----------------------------------------------------------------------------
// Operations
// ----------------------------------------------------------------------------

// Create empty total map with default value
pub open spec fn t_empty<V>(default: V) -> TotalMap<V> {
    TotalMap {
        default,
        map: Map::<nat, V>::empty(),
    }
}

// Get value (returns default if not present)
pub open spec fn t_get<V>(m: TotalMap<V>, k: nat) -> V {
    if m.map.dom().contains(k) {
        m.map[k]
    } else {
        m.default
    }
}

// Update value
pub open spec fn t_update<V>(m: TotalMap<V>, k: nat, v: V) -> TotalMap<V> {
    TotalMap {
        default: m.default,
        map: m.map.insert(k, v),
    }
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Get from empty returns default
pub proof fn t_get_empty<V>(default: V, k: nat)
    ensures t_get(t_empty(default), k) == default
{
}

// Get after update (same key)
pub proof fn t_get_update_eq<V>(m: TotalMap<V>, k: nat, v: V)
    ensures t_get(t_update(m, k, v), k) == v
{
}

// Get after update (different key)
pub proof fn t_get_update_neq<V>(m: TotalMap<V>, k1: nat, k2: nat, v: V)
    requires k1 != k2
    ensures t_get(t_update(m, k1, v), k2) == t_get(m, k2)
{
}

// Update shadow (last update wins)
pub proof fn t_update_shadow<V>(m: TotalMap<V>, k: nat, v1: V, v2: V)
    ensures t_get(t_update(t_update(m, k, v1), k, v2), k) == v2
{
}

// Update permutation (order doesn't matter for different keys)
pub proof fn t_update_permute<V>(m: TotalMap<V>, k1: nat, v1: V, k2: nat, v2: V)
    requires k1 != k2
    ensures forall|k: nat|
        t_get(t_update(t_update(m, k1, v1), k2, v2), k) ==
        t_get(t_update(t_update(m, k2, v2), k1, v1), k)
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_total_map()
{
    let m = t_empty(0nat);
    t_get_empty(0nat, 5);
    assert(t_get(m, 5) == 0);

    let m1 = t_update(m, 5, 10);
    t_get_update_eq(m, 5, 10nat);
    assert(t_get(m1, 5) == 10);

    t_get_update_neq(m, 5, 3, 10nat);
    assert(t_get(m1, 3) == 0);
}

pub proof fn example_shadow()
{
    let m = t_empty(0nat);
    let m1 = t_update(m, 1, 10);
    let m2 = t_update(m1, 1, 20);
    t_update_shadow(m, 1, 10nat, 20);
    assert(t_get(m2, 1) == 20);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn map_total_verify()
{
    example_total_map();
    example_shadow();

    let m = t_empty(0nat);
    t_get_empty(0nat, 42);

    let m1 = t_update(m, 10, 100);
    t_get_update_eq(m, 10, 100nat);
    t_get_update_neq(m, 10, 20, 100nat);
}

pub fn main() {
    proof {
        map_total_verify();
    }
}

} // verus!
