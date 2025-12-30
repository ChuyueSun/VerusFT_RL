use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Graph Paths (Supporting VFA)
// ============================================================================

pub struct Edge { pub src: nat, pub dst: nat }
pub struct Graph { pub edges: Seq<Edge>, pub n: nat }

pub open spec fn has_edge(g: Graph, u: nat, v: nat) -> bool {
    exists|i: int| #![auto] 0 <= i < g.edges.len() as int &&
        g.edges[i].src == u && g.edges[i].dst == v
}

// Path from u to v
pub open spec fn path(g: Graph, u: nat, v: nat, p: Seq<nat>) -> bool decreases p.len() {
    if p.len() == 0 { u == v }
    else { has_edge(g, u, p[0]) && path(g, p[0], v, p.skip(1)) }
}

pub open spec fn connected(g: Graph, u: nat, v: nat) -> bool {
    exists|p: Seq<nat>| path(g, u, v, p)
}

pub open spec fn reachable(g: Graph, u: nat) -> spec_fn(nat) -> bool {
    |v: nat| connected(g, u, v)
}

pub proof fn path_refl(g: Graph, u: nat)
    ensures path(g, u, u, Seq::empty())
{ reveal_with_fuel(path, 2); }

pub proof fn example_path() {
    path_refl(Graph { edges: Seq::empty(), n: 5 }, 0);
}

pub proof fn graph_path_verify() { example_path(); }
pub fn main() { proof { graph_path_verify(); } }

} // verus!
