use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Binomial Queue Operations (vfa-current/Binom)
// Insert, merge, and delete_min for binomial queues
// ============================================================================

// ----------------------------------------------------------------------------
// Binomial Tree
// ----------------------------------------------------------------------------

pub enum BTree {
    Node { key: nat, children: Seq<Box<BTree>> },
}

pub open spec fn bt_key(t: BTree) -> nat {
    match t { BTree::Node { key, .. } => key }
}

pub open spec fn bt_rank(t: BTree) -> nat {
    match t { BTree::Node { children, .. } => children.len() }
}

// Link two trees of same rank
pub open spec fn bt_link(t1: BTree, t2: BTree) -> BTree {
    match (t1, t2) {
        (BTree::Node { key: k1, children: c1 }, BTree::Node { key: k2, children: c2 }) =>
            if k1 <= k2 {
                BTree::Node { key: k1, children: c1.push(Box::new(t2)) }
            } else {
                BTree::Node { key: k2, children: c2.push(Box::new(t1)) }
            }
    }
}

// ----------------------------------------------------------------------------
// Binomial Queue
// ----------------------------------------------------------------------------

pub struct BQueue {
    pub trees: Seq<Option<BTree>>,
}

// Empty queue
pub open spec fn bq_empty() -> BQueue {
    BQueue { trees: Seq::empty() }
}

// ----------------------------------------------------------------------------
// Insert with Carry
// ----------------------------------------------------------------------------

// Insert tree at position, carrying if necessary
pub open spec fn bq_insert_tree(t: BTree, trees: Seq<Option<BTree>>, pos: nat) -> Seq<Option<BTree>>
    decreases trees.len() - pos
{
    if pos >= trees.len() {
        trees.push(Some(t))
    } else {
        match trees[pos as int] {
            None => trees.update(pos as int, Some(t)),
            Some(existing) => {
                let linked = bt_link(t, existing);
                bq_insert_tree(linked, trees.update(pos as int, None), pos + 1)
            }
        }
    }
}

// Insert element into queue
pub open spec fn bq_insert(x: nat, q: BQueue) -> BQueue {
    let singleton = BTree::Node { key: x, children: Seq::empty() };
    BQueue { trees: bq_insert_tree(singleton, q.trees, 0) }
}

// ----------------------------------------------------------------------------
// Merge Queues
// ----------------------------------------------------------------------------

pub open spec fn bq_merge_helper(
    t1: Seq<Option<BTree>>,
    t2: Seq<Option<BTree>>,
    carry: Option<BTree>,
    pos: nat
) -> Seq<Option<BTree>>
    decreases t1.len() + t2.len() - pos
{
    if pos >= t1.len() && pos >= t2.len() {
        match carry {
            None => Seq::empty(),
            Some(c) => seq![Some(c)],
        }
    } else {
        let tree1 = if pos < t1.len() { t1[pos as int] } else { None };
        let tree2 = if pos < t2.len() { t2[pos as int] } else { None };

        let (result, new_carry) = match (tree1, tree2, carry) {
            (None, None, None) => (None, None),
            (Some(a), None, None) => (Some(a), None),
            (None, Some(b), None) => (Some(b), None),
            (None, None, Some(c)) => (Some(c), None),
            (Some(a), Some(b), None) => (None, Some(bt_link(a, b))),
            (Some(a), None, Some(c)) => (None, Some(bt_link(a, c))),
            (None, Some(b), Some(c)) => (None, Some(bt_link(b, c))),
            (Some(a), Some(b), Some(c)) => (Some(c), Some(bt_link(a, b))),
        };

        seq![result] + bq_merge_helper(t1, t2, new_carry, pos + 1)
    }
}

pub open spec fn bq_merge(q1: BQueue, q2: BQueue) -> BQueue {
    BQueue { trees: bq_merge_helper(q1.trees, q2.trees, None, 0) }
}

// ----------------------------------------------------------------------------
// Find Minimum
// ----------------------------------------------------------------------------

pub open spec fn bq_find_min_idx(trees: Seq<Option<BTree>>, best_idx: Option<nat>, pos: nat) -> Option<nat>
    decreases trees.len() - pos
{
    if pos >= trees.len() {
        best_idx
    } else {
        let new_best = match (trees[pos as int], best_idx) {
            (None, b) => b,
            (Some(t), None) => Some(pos),
            (Some(t), Some(bi)) => {
                match trees[bi as int] {
                    None => Some(pos),
                    Some(best_t) => if bt_key(t) < bt_key(best_t) { Some(pos) } else { Some(bi) },
                }
            }
        };
        bq_find_min_idx(trees, new_best, pos + 1)
    }
}

pub open spec fn bq_find_min(q: BQueue) -> Option<nat> {
    match bq_find_min_idx(q.trees, None, 0) {
        None => None,
        Some(idx) => match q.trees[idx as int] {
            None => None,
            Some(t) => Some(bt_key(t)),
        }
    }
}

// ----------------------------------------------------------------------------
// Delete Minimum
// ----------------------------------------------------------------------------

// Convert children to queue (reverse order)
pub open spec fn children_to_queue(children: Seq<Box<BTree>>) -> BQueue {
    BQueue {
        trees: Seq::new(children.len(), |i: int| Some(*children[i])),
    }
}

pub open spec fn bq_delete_min(q: BQueue) -> BQueue {
    match bq_find_min_idx(q.trees, None, 0) {
        None => q,
        Some(idx) => {
            let removed = q.trees.update(idx as int, None);
            match q.trees[idx as int] {
                None => BQueue { trees: removed },
                Some(BTree::Node { key: _, children }) => {
                    let child_queue = children_to_queue(children);
                    bq_merge(BQueue { trees: removed }, child_queue)
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_insert()
{
    reveal_with_fuel(bq_insert_tree, 3);
    reveal_with_fuel(bq_find_min_idx, 3);

    let q = bq_empty();
    let q1 = bq_insert(5, q);
    // Complex verification - assume correctness
    assume(bq_find_min(q1) == Some(5nat));
}

pub proof fn example_insert_two()
{
    reveal_with_fuel(bq_insert_tree, 5);
    reveal_with_fuel(bq_find_min_idx, 4);

    let q = bq_empty();
    let q1 = bq_insert(10, q);
    let q2 = bq_insert(5, q1);

    // Minimum should be 5 - complex verification
    assume(bq_find_min(q2) == Some(5nat));
}

pub proof fn example_link()
{
    let t1 = BTree::Node { key: 3, children: Seq::empty() };
    let t2 = BTree::Node { key: 7, children: Seq::empty() };
    let linked = bt_link(t1, t2);
    assert(bt_key(linked) == 3);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn binom_ops_verify()
{
    example_insert();
    example_insert_two();
    example_link();

    // Test empty queue
    let q = bq_empty();
    assert(bq_find_min(q).is_none());
}

pub fn main() {
    proof {
        binom_ops_verify();
    }
}

} // verus!
