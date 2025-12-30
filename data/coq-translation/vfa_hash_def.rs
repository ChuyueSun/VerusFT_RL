use vstd::prelude::*;
use vstd::map::*;

verus! {

// ============================================================================
// VFA: Hash Table (Supporting VFA)
// ============================================================================

broadcast use group_map_axioms;

pub struct HashTable { pub buckets: Seq<Seq<(nat, nat)>>, pub size: nat }

pub open spec fn hash(k: nat, num_buckets: nat) -> nat recommends num_buckets > 0 { k % num_buckets }

pub open spec fn bucket_contains(bucket: Seq<(nat, nat)>, k: nat) -> bool decreases bucket.len() {
    bucket.len() > 0 && (bucket[0].0 == k || bucket_contains(bucket.skip(1), k))
}

pub open spec fn bucket_get(bucket: Seq<(nat, nat)>, k: nat) -> Option<nat> decreases bucket.len() {
    if bucket.len() == 0 { None }
    else if bucket[0].0 == k { Some(bucket[0].1) }
    else { bucket_get(bucket.skip(1), k) }
}

pub open spec fn ht_get(ht: HashTable, k: nat) -> Option<nat> recommends ht.size > 0 {
    bucket_get(ht.buckets[hash(k, ht.size) as int], k)
}

pub proof fn hash_bound(k: nat, n: nat) requires n > 0 ensures hash(k, n) < n {}

pub proof fn example_hash() {
    hash_bound(17, 5);
    assert(hash(17, 5) == 2);
}

pub proof fn hash_def_verify() { example_hash(); }
pub fn main() { proof { hash_def_verify(); } }

} // verus!
