use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: LRU Cache (Supporting VFA)
// ============================================================================

pub struct LRUCache { pub items: Seq<(nat, nat)>, pub capacity: nat }

pub open spec fn lru_empty(cap: nat) -> LRUCache { LRUCache { items: Seq::empty(), capacity: cap } }

pub open spec fn lru_get(cache: LRUCache, key: nat) -> Option<nat> decreases cache.items.len() {
    if cache.items.len() == 0 { None }
    else if cache.items[0].0 == key { Some(cache.items[0].1) }
    else { lru_get(LRUCache { items: cache.items.skip(1), capacity: cache.capacity }, key) }
}

pub open spec fn lru_contains(cache: LRUCache, key: nat) -> bool { lru_get(cache, key).is_some() }
pub open spec fn lru_size(cache: LRUCache) -> nat { cache.items.len() }
pub open spec fn lru_is_full(cache: LRUCache) -> bool { cache.items.len() >= cache.capacity }

pub proof fn empty_cache_empty(cap: nat) ensures !lru_contains(lru_empty(cap), 42) { reveal_with_fuel(lru_get, 2); }
pub proof fn empty_cache_size(cap: nat) ensures lru_size(lru_empty(cap)) == 0 {}

pub proof fn example_lru() { empty_cache_empty(10); empty_cache_size(10); }
pub proof fn lru_cache_verify() { example_lru(); }
pub fn main() { proof { lru_cache_verify(); } }

} // verus!
