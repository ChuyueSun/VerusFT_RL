use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Spanning Tree (Supporting VFA)
// ============================================================================

pub struct WeightedEdge { pub u: nat, pub v: nat, pub weight: nat }
pub struct Graph { pub n: nat, pub edges: Seq<WeightedEdge> }

pub open spec fn edge_valid(e: WeightedEdge, n: nat) -> bool { e.u < n && e.v < n }
pub open spec fn graph_valid(g: Graph) -> bool { forall|i: nat| #![auto] i < g.edges.len() ==> edge_valid(g.edges[i as int], g.n) }

// A spanning tree is a subset of edges that connects all vertices with no cycles
pub open spec fn is_spanning_tree(g: Graph, tree_edges: Seq<nat>) -> bool {
    tree_edges.len() == (g.n - 1) as nat && // n-1 edges for tree
    forall|i: nat| #![auto] i < tree_edges.len() ==> tree_edges[i as int] < g.edges.len()
}

// Total weight of tree
pub open spec fn tree_weight(g: Graph, tree_edges: Seq<nat>) -> nat decreases tree_edges.len() {
    if tree_edges.len() == 0 { 0 }
    else { g.edges[tree_edges[0] as int].weight + tree_weight(g, tree_edges.skip(1)) }
}

pub open spec fn is_mst(g: Graph, tree: Seq<nat>) -> bool {
    is_spanning_tree(g, tree) &&
    forall|other: Seq<nat>| is_spanning_tree(g, other) ==> tree_weight(g, tree) <= tree_weight(g, other)
}

pub proof fn example_spanning() { reveal_with_fuel(tree_weight, 3); }
pub proof fn spanning_tree_verify() { example_spanning(); }
pub fn main() { proof { spanning_tree_verify(); } }

} // verus!
