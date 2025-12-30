use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Dijkstra's Algorithm (Supporting VFA)
// ============================================================================

pub struct WeightedEdge { pub to: nat, pub weight: nat }
pub struct WeightedGraph { pub adj: Seq<Seq<WeightedEdge>>, pub n: nat }

pub open spec fn edge_valid(g: WeightedGraph, i: nat, j: nat) -> bool {
    g.adj[i as int][j as int].to < g.n
}

pub open spec fn wg_valid(g: WeightedGraph) -> bool {
    g.adj.len() == g.n && forall|i: nat, j: nat| #![trigger edge_valid(g, i, j)]
        i < g.n && j < g.adj[i as int].len() ==> edge_valid(g, i, j)
}

pub open spec fn distance_valid(dist: Seq<Option<nat>>, g: WeightedGraph, src: nat) -> bool {
    dist.len() == g.n && dist[src as int] == Some(0nat)
}

// Relaxation: if going through u to v is shorter, update v's distance
pub open spec fn relax(dist: Seq<Option<nat>>, u: nat, v: nat, w: nat) -> Seq<Option<nat>> {
    let new_dist = match dist[u as int] {
        None => dist[v as int],
        Some(du) => match dist[v as int] {
            None => Some(du + w),
            Some(dv) => if du + w < dv { Some(du + w) } else { Some(dv) }
        }
    };
    dist.update(v as int, new_dist)
}

pub proof fn relax_improves(dist: Seq<Option<nat>>, u: nat, v: nat, w: nat)
    requires u < dist.len(), v < dist.len()
    ensures match (dist[u as int], dist[v as int], relax(dist, u, v, w)[v as int]) {
        (Some(du), Some(dv), Some(new_dv)) => new_dv <= dv,
        _ => true,
    }
{}

pub proof fn dijkstra_verify() { }
pub fn main() { proof { dijkstra_verify(); } }

} // verus!
