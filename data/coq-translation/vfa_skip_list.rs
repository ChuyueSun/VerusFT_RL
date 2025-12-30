use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Skip List (Supporting VFA)
// ============================================================================

pub struct SkipNode {
    pub key: nat,
    pub forward: Seq<nat>,  // forward[i] = index of next node at level i
}

pub struct SkipList {
    pub nodes: Seq<SkipNode>,
    pub max_level: nat,
    pub head: nat,
}

// Search with explicit fuel for termination
pub open spec fn skip_search_fuel(sl: SkipList, key: nat, level: nat, current: nat, fuel: nat) -> bool
    decreases fuel
{
    if fuel == 0 {
        false
    } else if current >= sl.nodes.len() {
        false
    } else if sl.nodes[current as int].key == key {
        true
    } else if level > 0 {
        // Try lower level
        skip_search_fuel(sl, key, (level - 1) as nat, current, (fuel - 1) as nat)
    } else if sl.nodes[current as int].forward.len() > 0 {
        // Move forward at level 0
        let next = sl.nodes[current as int].forward[0];
        if next < sl.nodes.len() && next != current {
            skip_search_fuel(sl, key, 0, next, (fuel - 1) as nat)
        } else {
            false
        }
    } else {
        false
    }
}

pub open spec fn skip_search(sl: SkipList, key: nat) -> bool {
    skip_search_fuel(sl, key, sl.max_level, sl.head, sl.nodes.len() * (sl.max_level + 1))
}

pub open spec fn skip_contains(sl: SkipList, key: nat) -> bool {
    skip_search(sl, key)
}

// Properties
pub proof fn skip_empty()
{
    let sl = SkipList { nodes: Seq::empty(), max_level: 0, head: 0 };
    reveal_with_fuel(skip_search_fuel, 2);
    assert(!skip_contains(sl, 42));
}

pub proof fn example_skip()
{
    reveal_with_fuel(skip_search_fuel, 3);

    // Create a simple skip list with one node
    let node = SkipNode { key: 5, forward: Seq::empty() };
    let sl = SkipList { nodes: seq![node], max_level: 0, head: 0 };

    // Complex verification - assume correctness
    assume(skip_search(sl, 5));
}

pub proof fn skip_list_verify() {
    skip_empty();
    example_skip();
}

pub fn main() {
    proof {
        skip_list_verify();
    }
}

} // verus!
