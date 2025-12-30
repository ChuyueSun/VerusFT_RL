use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Dynamic Programming - Coin Change (Supporting VFA)
// Simplified to avoid termination issues
// ============================================================================

// Minimum coins to make amount (using index-based approach)
pub open spec fn min_coins(coins: Seq<nat>, amount: nat, i: nat) -> Option<nat>
    decreases coins.len() - i + amount
{
    if amount == 0 { Some(0) }
    else if i >= coins.len() { None }
    else {
        let skip = min_coins(coins, amount, i + 1);
        if coins[i as int] > amount || coins[i as int] == 0 { skip }
        else {
            let new_amount = (amount - coins[i as int]) as nat;
            // Recursive call with smaller amount
            let take = match min_coins(coins, new_amount, i) {
                None => None,
                Some(n) => Some(n + 1),
            };
            match (skip, take) {
                (None, t) => t,
                (s, None) => s,
                (Some(s), Some(t)) => if s < t { Some(s) } else { Some(t) },
            }
        }
    }
}

pub open spec fn solve_coins(coins: Seq<nat>, amount: nat) -> Option<nat> {
    min_coins(coins, amount, 0)
}

// Number of ways to make amount
pub open spec fn count_ways(coins: Seq<nat>, amount: nat, i: nat) -> nat
    decreases coins.len() - i + amount
{
    if amount == 0 { 1 }
    else if i >= coins.len() { 0 }
    else {
        let skip = count_ways(coins, amount, i + 1);
        if coins[i as int] > amount || coins[i as int] == 0 { skip }
        else {
            let new_amount = (amount - coins[i as int]) as nat;
            skip + count_ways(coins, new_amount, i)
        }
    }
}

pub proof fn zero_amount(coins: Seq<nat>)
    ensures solve_coins(coins, 0) == Some(0nat)
{
    reveal_with_fuel(min_coins, 2);
}

pub proof fn one_way_zero(coins: Seq<nat>)
    ensures count_ways(coins, 0, 0) == 1
{
    reveal_with_fuel(count_ways, 2);
}

pub proof fn dp_coins_verify() {
    zero_amount(seq![1nat, 5, 10]);
    one_way_zero(seq![1nat, 5, 10]);
}

pub fn main() { proof { dp_coins_verify(); } }

} // verus!
