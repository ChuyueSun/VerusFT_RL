use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Suffix Array (Supporting VFA)
// ============================================================================

pub struct SuffixArray {
    pub text: Seq<nat>,
    pub sa: Seq<nat>,
}

pub open spec fn suffix(text: Seq<nat>, start: nat) -> Seq<nat> {
    if start <= text.len() {
        text.skip(start as int)
    } else {
        Seq::empty()
    }
}

pub open spec fn sa_valid(sa: SuffixArray) -> bool {
    sa.sa.len() == sa.text.len() &&
    forall|i: nat| #![auto] i < sa.sa.len() ==> sa.sa[i as int] < sa.text.len()
}

// Lexicographic comparison with fuel for termination
pub open spec fn suffix_le_fuel(text: Seq<nat>, i: nat, j: nat, fuel: nat) -> bool
    decreases fuel
{
    if fuel == 0 {
        true  // Default to true for termination
    } else {
        let si = suffix(text, i);
        let sj = suffix(text, j);
        if si.len() == 0 {
            true
        } else if sj.len() == 0 {
            false
        } else if si[0] < sj[0] {
            true
        } else if si[0] > sj[0] {
            false
        } else {
            suffix_le_fuel(text, i + 1, j + 1, (fuel - 1) as nat)
        }
    }
}

pub open spec fn suffix_le(text: Seq<nat>, i: nat, j: nat) -> bool {
    suffix_le_fuel(text, i, j, text.len())
}

pub open spec fn sa_pair_sorted(sa: SuffixArray, i: nat) -> bool {
    suffix_le(sa.text, sa.sa[i as int], sa.sa[(i + 1) as int])
}

pub open spec fn sa_sorted(sa: SuffixArray) -> bool {
    forall|i: nat| #[trigger] sa_pair_sorted(sa, i) || !(i + 1 < sa.sa.len()) || sa_pair_sorted(sa, i)
}

// Properties
pub proof fn empty_suffix_array()
{
    let sa = SuffixArray { text: Seq::empty(), sa: Seq::empty() };
    assert(sa_valid(sa));
}

pub proof fn single_char_suffix_array()
{
    let sa = SuffixArray { text: seq![5nat], sa: seq![0nat] };
    assert(sa_valid(sa));
}

pub proof fn example_suffix()
{
    reveal_with_fuel(suffix_le_fuel, 3);

    let text = seq![1nat, 2, 3];
    // suffix(text, 0) = [1, 2, 3]
    // suffix(text, 1) = [2, 3]
    // suffix(text, 2) = [3]
    assert(suffix(text, 0).len() == 3);
    assert(suffix(text, 1).len() == 2);
    assert(suffix(text, 2).len() == 1);
}

pub proof fn suffix_array_verify() {
    empty_suffix_array();
    single_char_suffix_array();
    example_suffix();
}

pub fn main() {
    proof {
        suffix_array_verify();
    }
}

} // verus!
