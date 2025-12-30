use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// VFA: Graph Definition (Supporting VFA)
// ============================================================================

broadcast use group_map_axioms;

pub struct Graph { pub adj: Map<nat, Seq<nat>>, pub n: nat }

pub open spec fn has_edge(g: Graph, u: nat, v: nat) -> bool {
    g.adj.dom().contains(u) && seq_mem(g.adj[u], v)
}

pub open spec fn seq_mem(s: Seq<nat>, x: nat) -> bool decreases s.len() {
    s.len() > 0 && (s[0] == x || seq_mem(s.skip(1), x))
}

pub open spec fn degree(g: Graph, v: nat) -> nat {
    if g.adj.dom().contains(v) { g.adj[v].len() } else { 0 }
}

pub open spec fn is_undirected(g: Graph) -> bool {
    forall|u: nat, v: nat| has_edge(g, u, v) ==> has_edge(g, v, u)
}

pub proof fn example_graph() {
    let adj = Map::empty().insert(0nat, seq![1nat]).insert(1nat, seq![0nat]);
    let g = Graph { adj, n: 2 };
    reveal_with_fuel(seq_mem, 2);
    assert(has_edge(g, 0, 1));
}

pub proof fn graph_def_verify() { example_graph(); }
pub fn main() { proof { graph_def_verify(); } }

} // verus!
