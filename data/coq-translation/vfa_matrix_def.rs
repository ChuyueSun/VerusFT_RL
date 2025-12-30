use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Matrix Definition (Supporting VFA)
// ============================================================================

pub struct Matrix { pub rows: nat, pub cols: nat, pub data: Seq<Seq<nat>> }

pub open spec fn valid_matrix(m: Matrix) -> bool {
    m.data.len() == m.rows && forall|i: nat| #![auto] i < m.rows ==> m.data[i as int].len() == m.cols
}

pub open spec fn get(m: Matrix, i: nat, j: nat) -> nat
    recommends i < m.rows, j < m.cols, valid_matrix(m)
{ m.data[i as int][j as int] }

pub open spec fn zero_matrix(r: nat, c: nat) -> Matrix {
    Matrix { rows: r, cols: c, data: Seq::new(r, |_i: int| Seq::new(c, |_j: int| 0nat)) }
}

pub open spec fn identity(n: nat) -> Matrix {
    Matrix { rows: n, cols: n, data: Seq::new(n, |i: int| Seq::new(n, |j: int| if i == j { 1nat } else { 0nat })) }
}

pub proof fn zero_valid(r: nat, c: nat) ensures valid_matrix(zero_matrix(r, c)) {}
pub proof fn identity_valid(n: nat) ensures valid_matrix(identity(n)) {}

pub proof fn example_matrix() {
    zero_valid(3, 3);
    identity_valid(4);
    let m = identity(3);
    assert(get(m, 0, 0) == 1);
    assert(get(m, 0, 1) == 0);
}

pub proof fn matrix_def_verify() { example_matrix(); }
pub fn main() { proof { matrix_def_verify(); } }

} // verus!
