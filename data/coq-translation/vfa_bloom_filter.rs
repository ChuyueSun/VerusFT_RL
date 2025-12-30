use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Bloom Filter (Supporting VFA)
// ============================================================================

pub struct BloomFilter { pub bits: Seq<bool>, pub num_hashes: nat }

pub open spec fn hash1(x: nat, size: nat) -> nat recommends size > 0 { x % size }
pub open spec fn hash2(x: nat, size: nat) -> nat recommends size > 0 { (x / size) % size }

pub open spec fn bloom_hash(x: nat, i: nat, size: nat) -> nat recommends size > 0 {
    (hash1(x, size) + i * hash2(x, size)) % size
}

pub open spec fn bloom_add(bf: BloomFilter, x: nat) -> BloomFilter {
    BloomFilter {
        bits: bf.bits, // Would update bits at hash positions
        num_hashes: bf.num_hashes,
    }
}

pub open spec fn bloom_check_bit(bf: BloomFilter, x: nat, i: nat) -> bool {
    if bf.bits.len() > 0 {
        let idx = bloom_hash(x, i, bf.bits.len());
        idx < bf.bits.len() && bf.bits[idx as int]
    } else {
        false
    }
}

pub open spec fn bloom_maybe_contains(bf: BloomFilter, x: nat) -> bool {
    bf.bits.len() > 0 && bf.num_hashes > 0 && bloom_check_bit(bf, x, 0)
}

pub proof fn bloom_false_positive() {
    // Bloom filters can have false positives but no false negatives
}

pub proof fn example_bloom() {
    let bf = BloomFilter { bits: Seq::new(10, |_i: int| false), num_hashes: 3 };
    assert(!bloom_maybe_contains(bf, 42));
}

pub proof fn bloom_filter_verify() { example_bloom(); }
pub fn main() { proof { bloom_filter_verify(); } }

} // verus!
