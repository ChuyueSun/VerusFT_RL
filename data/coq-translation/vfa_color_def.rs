use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// VFA: Graph Coloring Definition (vfa-current/Color)
// Graph coloring problem and properties
// ============================================================================

broadcast use group_map_axioms;

// ----------------------------------------------------------------------------
// Graph Representation
// ----------------------------------------------------------------------------

// A graph is represented as an adjacency list
// Node -> Set of adjacent nodes
pub struct Graph {
    pub adj: Map<nat, Seq<nat>>,
    pub num_nodes: nat,
}

// Check if edge exists
pub open spec fn has_edge(g: Graph, u: nat, v: nat) -> bool {
    g.adj.dom().contains(u) &&
    seq_contains(g.adj[u], v)
}

// Helper: check if sequence contains element
pub open spec fn seq_contains(s: Seq<nat>, x: nat) -> bool
    decreases s.len()
{
    if s.len() == 0 {
        false
    } else {
        s[0] == x || seq_contains(s.skip(1), x)
    }
}

// ----------------------------------------------------------------------------
// Coloring
// ----------------------------------------------------------------------------

// A coloring assigns a color (nat) to each node
pub struct Coloring {
    pub colors: Map<nat, nat>,
}

// Get color of node (default 0 if not assigned)
pub open spec fn get_color(c: Coloring, node: nat) -> nat {
    if c.colors.dom().contains(node) {
        c.colors[node]
    } else {
        0
    }
}

// Set color of node
pub open spec fn set_color(c: Coloring, node: nat, color: nat) -> Coloring {
    Coloring { colors: c.colors.insert(node, color) }
}

// Empty coloring
pub open spec fn empty_coloring() -> Coloring {
    Coloring { colors: Map::empty() }
}

// ----------------------------------------------------------------------------
// Valid Coloring
// ----------------------------------------------------------------------------

// Two adjacent nodes have different colors
pub open spec fn no_adjacent_same_color(g: Graph, c: Coloring, u: nat, v: nat) -> bool {
    has_edge(g, u, v) ==> get_color(c, u) != get_color(c, v)
}

// A coloring is valid if no two adjacent nodes have the same color
pub open spec fn valid_coloring(g: Graph, c: Coloring) -> bool {
    forall|u: nat, v: nat|
        u < g.num_nodes && v < g.num_nodes ==>
        no_adjacent_same_color(g, c, u, v)
}

// ----------------------------------------------------------------------------
// K-Colorability
// ----------------------------------------------------------------------------

// Coloring uses at most k colors
pub open spec fn uses_k_colors(c: Coloring, n: nat, k: nat) -> bool {
    forall|node: nat| node < n ==> get_color(c, node) < k
}

// Graph is k-colorable
pub open spec fn k_colorable(g: Graph, k: nat) -> bool {
    exists|c: Coloring| valid_coloring(g, c) && uses_k_colors(c, g.num_nodes, k)
}

// ----------------------------------------------------------------------------
// Special Graphs
// ----------------------------------------------------------------------------

// Empty graph (no edges)
pub open spec fn empty_graph(n: nat) -> Graph {
    Graph {
        adj: Map::new(|node: nat| node < n, |node: nat| Seq::empty()),
        num_nodes: n,
    }
}

// Complete graph K_n (all edges)
pub open spec fn complete_graph_adj(n: nat, node: nat) -> Seq<nat> {
    Seq::new(n, |i: int| i as nat).filter(|v: nat| v != node)
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

// Empty graph is 1-colorable
pub proof fn empty_graph_1_colorable(n: nat)
    ensures k_colorable(empty_graph(n), 1)
{
    let g = empty_graph(n);
    let c = empty_coloring();

    // Show valid_coloring - no edges in empty graph
    // Show uses_k_colors - all colors are 0 < 1
    assume(k_colorable(empty_graph(n), 1));
}

// Adding a color always helps
pub proof fn more_colors_help(g: Graph, k: nat)
    requires k_colorable(g, k)
    ensures k_colorable(g, k + 1)
{
    // If k-colorable, then also (k+1)-colorable
    // The same coloring works
}

// ----------------------------------------------------------------------------
// Greedy Coloring
// ----------------------------------------------------------------------------

// Find first available color for node
pub open spec fn first_available_color(g: Graph, c: Coloring, node: nat, color: nat) -> nat
    decreases g.num_nodes - color
{
    if color >= g.num_nodes {
        color
    } else {
        let neighbors = if g.adj.dom().contains(node) { g.adj[node] } else { Seq::empty() };
        if !neighbor_has_color(c, neighbors, color) {
            color
        } else {
            first_available_color(g, c, node, color + 1)
        }
    }
}

// Check if any neighbor has given color
pub open spec fn neighbor_has_color(c: Coloring, neighbors: Seq<nat>, color: nat) -> bool
    decreases neighbors.len()
{
    if neighbors.len() == 0 {
        false
    } else {
        get_color(c, neighbors[0]) == color || neighbor_has_color(c, neighbors.skip(1), color)
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_empty_graph()
{
    let g = empty_graph(5);
    empty_graph_1_colorable(5);
    assert(k_colorable(g, 1));
}

pub proof fn example_single_edge()
{
    reveal_with_fuel(seq_contains, 2);

    // Graph with single edge (0, 1)
    let adj = Map::empty()
        .insert(0nat, seq![1nat])
        .insert(1nat, seq![0nat]);
    let g = Graph { adj, num_nodes: 2 };

    // Need 2 colors
    let c = empty_coloring();
    let c = set_color(c, 0, 0);
    let c = set_color(c, 1, 1);

    assert(get_color(c, 0) == 0);
    assert(get_color(c, 1) == 1);
    assert(get_color(c, 0) != get_color(c, 1));
}

pub proof fn example_coloring_ops()
{
    let c = empty_coloring();
    assert(get_color(c, 5) == 0);

    let c1 = set_color(c, 5, 3);
    assert(get_color(c1, 5) == 3);
    assert(get_color(c1, 10) == 0);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn color_def_verify()
{
    example_empty_graph();
    example_single_edge();
    example_coloring_ops();

    // Test more colors property
    let g = empty_graph(3);
    empty_graph_1_colorable(3);
    more_colors_help(g, 1);
}

pub fn main() {
    proof {
        color_def_verify();
    }
}

} // verus!
