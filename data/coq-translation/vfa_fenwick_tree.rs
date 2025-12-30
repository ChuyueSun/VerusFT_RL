use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Fenwick Tree / Binary Indexed Tree (Supporting VFA)
// Simplified implementation without bitwise operations on nat
// ============================================================================

pub struct FenwickTree { pub tree: Seq<nat>, pub n: nat }

// Simplified: just use increments of 1 for tree traversal
// In real implementation, lowbit would use bit manipulation
pub open spec fn parent_index(i: nat) -> nat
    decreases i
{
    if i <= 1 { 0 } else { (i / 2) as nat }
}

pub open spec fn prefix_sum(ft: FenwickTree, i: nat) -> nat
    decreases i
{
    if i == 0 { 0 }
    else if i > ft.n { 0 }
    else { ft.tree[(i - 1) as int] + prefix_sum(ft, (i - 1) as nat) }
}

pub open spec fn range_sum(ft: FenwickTree, l: nat, r: nat) -> nat
    recommends l <= r
{
    if l == 0 { prefix_sum(ft, r) }
    else if prefix_sum(ft, r) >= prefix_sum(ft, l) {
        (prefix_sum(ft, r) - prefix_sum(ft, l)) as nat
    } else { 0 }
}

pub open spec fn ft_init(n: nat) -> FenwickTree {
    FenwickTree { tree: Seq::new(n, |_i: int| 0nat), n }
}

pub proof fn ft_init_zero(n: nat)
    ensures prefix_sum(ft_init(n), 0) == 0
{
    reveal_with_fuel(prefix_sum, 2);
}

pub proof fn prefix_sum_empty()
    ensures prefix_sum(ft_init(0), 0) == 0
{
    reveal_with_fuel(prefix_sum, 2);
}

pub proof fn example_fenwick() {
    reveal_with_fuel(prefix_sum, 3);
    ft_init_zero(10);
    prefix_sum_empty();
}

pub proof fn fenwick_tree_verify() { example_fenwick(); }
pub fn main() { proof { fenwick_tree_verify(); } }

} // verus!
