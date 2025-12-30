use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Bellman-Ford Algorithm (Supporting VFA)
// ============================================================================

pub struct Edge { pub from: nat, pub to: nat, pub weight: int }
pub struct Graph { pub edges: Seq<Edge>, pub n: nat }

pub open spec fn graph_valid(g: Graph) -> bool {
    forall|i: nat| #![auto] i < g.edges.len() ==> g.edges[i as int].from < g.n && g.edges[i as int].to < g.n
}

// Relax all edges once
pub open spec fn relax_all(dist: Seq<Option<int>>, edges: Seq<Edge>, i: nat) -> Seq<Option<int>>
    decreases edges.len() - i
{
    if i >= edges.len() { dist }
    else {
        let e = edges[i as int];
        let new_dist = match dist[e.from as int] {
            None => dist,
            Some(du) => match dist[e.to as int] {
                None => dist.update(e.to as int, Some(du + e.weight)),
                Some(dv) => if du + e.weight < dv { dist.update(e.to as int, Some(du + e.weight)) } else { dist }
            }
        };
        relax_all(new_dist, edges, i + 1)
    }
}

// Detect negative cycle: if relaxation improves any distance, negative cycle exists
pub open spec fn has_negative_cycle(dist: Seq<Option<int>>, edges: Seq<Edge>) -> bool {
    relax_all(dist, edges, 0) != dist
}

pub proof fn example_bellman_ford() { reveal_with_fuel(relax_all, 3); }
pub proof fn bellman_ford_verify() { example_bellman_ford(); }
pub fn main() { proof { bellman_ford_verify(); } }

} // verus!
