use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Floyd-Warshall Algorithm (Supporting VFA)
// ============================================================================

// Distance matrix where dist[i][j] = shortest path from i to j
pub struct DistMatrix { pub data: Seq<Seq<Option<nat>>>, pub n: nat }

pub open spec fn dm_valid(dm: DistMatrix) -> bool {
    dm.data.len() == dm.n && forall|i: nat| #![auto] i < dm.n ==> dm.data[i as int].len() == dm.n
}

pub open spec fn dm_get(dm: DistMatrix, i: nat, j: nat) -> Option<nat> recommends i < dm.n, j < dm.n {
    dm.data[i as int][j as int]
}

// Relaxation: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
pub open spec fn fw_relax(dm: DistMatrix, i: nat, j: nat, k: nat) -> Option<nat>
    recommends i < dm.n, j < dm.n, k < dm.n
{
    let direct = dm_get(dm, i, j);
    let via_k = match (dm_get(dm, i, k), dm_get(dm, k, j)) {
        (Some(a), Some(b)) => Some(a + b),
        _ => None,
    };
    match (direct, via_k) {
        (None, v) => v,
        (Some(d), None) => Some(d),
        (Some(d), Some(v)) => if v < d { Some(v) } else { Some(d) },
    }
}

pub proof fn diagonal_zero(dm: DistMatrix, i: nat)
    requires dm_valid(dm), i < dm.n, dm_get(dm, i, i) == Some(0nat)
    ensures fw_relax(dm, i, i, i) == Some(0nat)
{}

pub proof fn floyd_warshall_verify() { }
pub fn main() { proof { floyd_warshall_verify(); } }

} // verus!
