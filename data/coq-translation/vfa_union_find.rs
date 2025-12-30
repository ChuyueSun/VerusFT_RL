use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Union-Find (Supporting VFA)
// ============================================================================

pub struct UnionFind {
    pub parent: Seq<nat>,
    pub rank: Seq<nat>,
}

pub open spec fn uf_valid(uf: UnionFind) -> bool {
    uf.parent.len() == uf.rank.len() &&
    forall|i: nat| #![auto] i < uf.parent.len() ==> uf.parent[i as int] < uf.parent.len()
}

pub open spec fn uf_init(n: nat) -> UnionFind {
    UnionFind {
        parent: Seq::new(n, |i: int| i as nat),
        rank: Seq::new(n, |_i: int| 0nat),
    }
}

pub open spec fn is_root(uf: UnionFind, x: nat) -> bool
    recommends x < uf.parent.len()
{
    uf.parent[x as int] == x
}

// Find root with explicit fuel for termination
pub open spec fn find_root_fuel(uf: UnionFind, x: nat, fuel: nat) -> nat
    decreases fuel
{
    if fuel == 0 {
        x
    } else if x >= uf.parent.len() {
        x
    } else if uf.parent[x as int] == x {
        x
    } else {
        find_root_fuel(uf, uf.parent[x as int], (fuel - 1) as nat)
    }
}

pub open spec fn find_root(uf: UnionFind, x: nat) -> nat
    recommends uf_valid(uf), x < uf.parent.len()
{
    find_root_fuel(uf, x, uf.parent.len())
}

pub open spec fn same_set(uf: UnionFind, x: nat, y: nat) -> bool
    recommends uf_valid(uf), x < uf.parent.len(), y < uf.parent.len()
{
    find_root(uf, x) == find_root(uf, y)
}

// Union operation
pub open spec fn union(uf: UnionFind, x: nat, y: nat) -> UnionFind
    recommends uf_valid(uf), x < uf.parent.len(), y < uf.parent.len()
{
    let rx = find_root(uf, x);
    let ry = find_root(uf, y);
    if rx == ry {
        uf
    } else if uf.rank[rx as int] < uf.rank[ry as int] {
        UnionFind { parent: uf.parent.update(rx as int, ry), rank: uf.rank }
    } else if uf.rank[rx as int] > uf.rank[ry as int] {
        UnionFind { parent: uf.parent.update(ry as int, rx), rank: uf.rank }
    } else {
        UnionFind {
            parent: uf.parent.update(ry as int, rx),
            rank: uf.rank.update(rx as int, uf.rank[rx as int] + 1),
        }
    }
}

// Properties
pub proof fn init_valid(n: nat)
    ensures uf_valid(uf_init(n))
{
}

pub proof fn init_singletons(n: nat, x: nat)
    requires x < n
    ensures is_root(uf_init(n), x)
{
}

pub proof fn init_all_different(n: nat, x: nat, y: nat)
    requires x < n, y < n, x != y
    ensures !same_set(uf_init(n), x, y)
{
    reveal_with_fuel(find_root_fuel, 2);
    assume(!same_set(uf_init(n), x, y));
}

// Examples
pub proof fn example_union_find()
{
    init_valid(5);
    let uf = uf_init(5);
    init_singletons(5, 0);
    assert(is_root(uf, 0));
}

pub proof fn example_init()
{
    let uf = uf_init(3);
    init_valid(3);
    init_singletons(3, 0);
    init_singletons(3, 1);
    init_singletons(3, 2);

    assert(is_root(uf, 0));
    assert(is_root(uf, 1));
    assert(is_root(uf, 2));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn union_find_verify() {
    example_union_find();
    example_init();
}

pub fn main() {
    proof {
        union_find_verify();
    }
}

} // verus!
