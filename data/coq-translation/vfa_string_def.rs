use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: String Operations (Supporting VFA)
// Strings as sequences of characters (nat for simplicity)
// ============================================================================

pub type Str = Seq<nat>;

pub open spec fn str_len(s: Str) -> nat { s.len() }
pub open spec fn str_empty() -> Str { Seq::empty() }
pub open spec fn str_concat(s1: Str, s2: Str) -> Str { s1 + s2 }

pub open spec fn is_prefix(pre: Str, s: Str) -> bool {
    pre.len() <= s.len() && s.take(pre.len() as int) =~= pre
}

pub open spec fn is_suffix(suf: Str, s: Str) -> bool {
    suf.len() <= s.len() && s.skip((s.len() - suf.len()) as int) =~= suf
}

pub open spec fn is_substring(sub: Str, s: Str) -> bool {
    exists|i: nat| i + sub.len() <= s.len() && #[trigger] s.subrange(i as int, (i + sub.len()) as int) =~= sub
}

pub proof fn empty_prefix(s: Str) ensures is_prefix(str_empty(), s) {}
pub proof fn empty_suffix(s: Str) ensures is_suffix(str_empty(), s) {}
pub proof fn self_prefix(s: Str) ensures is_prefix(s, s) {}
pub proof fn self_suffix(s: Str) ensures is_suffix(s, s) {}

pub proof fn concat_len(s1: Str, s2: Str) ensures str_len(str_concat(s1, s2)) == str_len(s1) + str_len(s2) {}

pub proof fn example_string() {
    let s = seq![1nat, 2, 3, 4, 5];
    empty_prefix(s);
    self_prefix(s);
    concat_len(seq![1nat, 2], seq![3nat, 4, 5]);
}

pub proof fn string_def_verify() { example_string(); }
pub fn main() { proof { string_def_verify(); } }

} // verus!
