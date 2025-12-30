use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Topological Sort (Supporting VFA)
// ============================================================================

pub struct DAG { pub adj: Seq<Seq<nat>>, pub n: nat }

pub open spec fn edge_in_range(g: DAG, i: nat, j: nat) -> bool {
    g.adj[i as int][j as int] < g.n
}

pub open spec fn dag_valid(g: DAG) -> bool {
    g.adj.len() == g.n && forall|i: nat, j: nat| #![trigger edge_in_range(g, i, j)]
        i < g.n && j < g.adj[i as int].len() ==> edge_in_range(g, i, j)
}

pub open spec fn has_edge(g: DAG, u: nat, v: nat) -> bool recommends u < g.n {
    u < g.adj.len() && exists|i: nat| #![auto] i < g.adj[u as int].len() && g.adj[u as int][i as int] == v
}

// A valid topological order: if (u,v) is edge, u appears before v
pub open spec fn is_topo_order(g: DAG, order: Seq<nat>) -> bool {
    order.len() == g.n &&
    forall|u: nat, v: nat| has_edge(g, u, v) ==> {
        exists|i: nat, j: nat| #![auto] i < order.len() && j < order.len() && order[i as int] == u && order[j as int] == v && i < j
    }
}

pub proof fn empty_dag_topo() {
    let g = DAG { adj: Seq::empty(), n: 0 };
    assert(is_topo_order(g, Seq::empty()));
}

pub proof fn topological_sort_verify() { empty_dag_topo(); }
pub fn main() { proof { topological_sort_verify(); } }

} // verus!
