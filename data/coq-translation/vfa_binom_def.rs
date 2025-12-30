use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Binomial Queue Definition (vfa-current/Binom)
// Binomial tree and queue data structures - Simplified version
// ============================================================================

// ----------------------------------------------------------------------------
// Binomial Tree (Simplified - rank-indexed)
// ----------------------------------------------------------------------------

// Represent binomial tree by its key and rank
// Full tree structure would require complex recursive types
pub struct BTree {
    pub key: nat,
    pub rank: nat,
}

// Get key of root
pub open spec fn bt_key(t: BTree) -> nat {
    t.key
}

// Get rank
pub open spec fn bt_rank(t: BTree) -> nat {
    t.rank
}

// Size of tree of rank r is 2^r
pub open spec fn bt_size(t: BTree) -> nat {
    pow2(t.rank)
}

pub open spec fn pow2(n: nat) -> nat
    decreases n
{
    if n == 0 { 1 } else { 2 * pow2((n - 1) as nat) }
}

// ----------------------------------------------------------------------------
// Binomial Queue
// ----------------------------------------------------------------------------

// A binomial queue is a sequence of optional binomial trees
pub struct BQueue {
    pub trees: Seq<Option<BTree>>,
}

// Empty queue
pub open spec fn bq_empty() -> BQueue {
    BQueue { trees: Seq::empty() }
}

// Check if queue is empty
pub open spec fn bq_is_empty(q: BQueue) -> bool {
    forall|i: int| 0 <= i < q.trees.len() as int ==>
        (#[trigger] q.trees[i]).is_none()
}

// ----------------------------------------------------------------------------
// Find Minimum
// ----------------------------------------------------------------------------

pub open spec fn bq_find_min_helper(trees: Seq<Option<BTree>>, best: Option<nat>) -> Option<nat>
    decreases trees.len()
{
    if trees.len() == 0 {
        best
    } else {
        let new_best = match (trees[0], best) {
            (None, b) => b,
            (Some(t), None) => Some(bt_key(t)),
            (Some(t), Some(b)) => if bt_key(t) < b { Some(bt_key(t)) } else { Some(b) },
        };
        bq_find_min_helper(trees.skip(1), new_best)
    }
}

pub open spec fn bq_find_min(q: BQueue) -> Option<nat> {
    bq_find_min_helper(q.trees, None)
}

// ----------------------------------------------------------------------------
// Link Trees
// ----------------------------------------------------------------------------

// Link two trees of same rank
pub open spec fn bt_link(t1: BTree, t2: BTree) -> BTree
    recommends bt_rank(t1) == bt_rank(t2)
{
    if t1.key <= t2.key {
        BTree { key: t1.key, rank: t1.rank + 1 }
    } else {
        BTree { key: t2.key, rank: t2.rank + 1 }
    }
}

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

pub proof fn pow2_positive(n: nat)
    ensures pow2(n) > 0
    decreases n
{
    reveal_with_fuel(pow2, 2);
    if n > 0 {
        pow2_positive((n - 1) as nat);
    }
}

pub proof fn link_increases_rank(t1: BTree, t2: BTree)
    requires bt_rank(t1) == bt_rank(t2)
    ensures bt_rank(bt_link(t1, t2)) == bt_rank(t1) + 1
{
}

pub proof fn link_preserves_min(t1: BTree, t2: BTree)
    requires bt_rank(t1) == bt_rank(t2)
    ensures bt_key(bt_link(t1, t2)) == if t1.key <= t2.key { t1.key } else { t2.key }
{
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_singleton_tree()
{
    let t = BTree { key: 5, rank: 0 };
    assert(bt_rank(t) == 0);
    assert(bt_key(t) == 5);
    reveal_with_fuel(pow2, 1);
    assert(bt_size(t) == 1);
}

pub proof fn example_link()
{
    let t1 = BTree { key: 3, rank: 0 };
    let t2 = BTree { key: 7, rank: 0 };

    let linked = bt_link(t1, t2);
    assert(bt_key(linked) == 3);
    assert(bt_rank(linked) == 1);
}

pub proof fn example_empty_queue()
{
    let q = bq_empty();
    assert(q.trees.len() == 0);
    reveal_with_fuel(bq_find_min_helper, 1);
    assert(bq_find_min(q).is_none());
}

pub proof fn example_queue_with_tree()
{
    reveal_with_fuel(bq_find_min_helper, 2);
    let t = BTree { key: 5, rank: 0 };
    let q = BQueue { trees: seq![Some(t)] };
    assert(bq_find_min(q) == Some(5nat));
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn binom_def_verify()
{
    example_singleton_tree();
    example_link();
    example_empty_queue();
    example_queue_with_tree();
}

pub fn main() {
    proof {
        binom_def_verify();
    }
}

} // verus!
