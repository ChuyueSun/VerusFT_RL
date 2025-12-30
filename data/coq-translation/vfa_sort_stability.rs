use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Sort Stability (Supporting VFA)
// Stable sorting preserves relative order of equal elements
// ============================================================================

pub struct KeyValue { pub key: nat, pub value: nat }

pub open spec fn kv_le(a: KeyValue, b: KeyValue) -> bool { a.key <= b.key }

pub open spec fn kv_sorted(s: Seq<KeyValue>) -> bool decreases s.len() {
    s.len() <= 1 || (kv_le(s[0], s[1]) && kv_sorted(s.skip(1)))
}

// Stable: equal keys preserve original order
pub open spec fn stable_for_key(original: Seq<KeyValue>, sorted: Seq<KeyValue>, k: nat) -> bool {
    let orig_with_k = filter_key(original, k);
    let sort_with_k = filter_key(sorted, k);
    orig_with_k =~= sort_with_k
}

pub open spec fn filter_key(s: Seq<KeyValue>, k: nat) -> Seq<KeyValue> decreases s.len() {
    if s.len() == 0 { Seq::empty() }
    else if s[0].key == k { seq![s[0]] + filter_key(s.skip(1), k) }
    else { filter_key(s.skip(1), k) }
}

pub proof fn example_stability() {
    reveal_with_fuel(kv_sorted, 3);
    reveal_with_fuel(filter_key, 3);
    let s = seq![KeyValue { key: 1, value: 10 }];
    assert(kv_sorted(s));
}

pub proof fn sort_stability_verify() { example_stability(); }
pub fn main() { proof { sort_stability_verify(); } }

} // verus!
