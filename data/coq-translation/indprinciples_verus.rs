use vstd::prelude::*;

verus! {

// IndPrinciples (Software Foundations, LF/IndPrinciples) style snippets in Verus.
// Focus: picking the right induction principle / variable, and strengthening statements.

// ----------------------------
// Basic arithmetic on nat
// ----------------------------

pub open spec fn add(a: nat, b: nat) -> nat
    decreases a
{
    if a == 0 { b } else { add((a - 1) as nat, b) + 1 }
}

pub open spec fn mul(a: nat, b: nat) -> nat
    decreases a
{
    if a == 0 { 0 } else { add(b, mul((a - 1) as nat, b)) }
}

pub proof fn lemma_add_0_l(n: nat)
    ensures add(0, n) == n
{
}

pub proof fn lemma_add_0_r(n: nat)
    ensures add(n, 0) == n
    decreases n
{
    if n == 0 {
    } else {
        lemma_add_0_r((n - 1) as nat);
    }
}

pub proof fn lemma_add_succ_r(n: nat, m: nat)
    ensures add(n, m + 1) == add(n, m) + 1
    decreases n
{
    if n == 0 {
    } else {
        lemma_add_succ_r((n - 1) as nat, m);
    }
}

pub proof fn lemma_add_assoc(a: nat, b: nat, c: nat)
    ensures add(add(a, b), c) == add(a, add(b, c))
    decreases a
{
    if a == 0 {
    } else {
        lemma_add_assoc((a - 1) as nat, b, c);
    }
}

// ----------------------------
// Strengthening the induction hypothesis
// ----------------------------

// Classic: commutativity is easiest when we generalize over the other argument.
pub proof fn lemma_add_comm(n: nat, m: nat)
    ensures add(n, m) == add(m, n)
    decreases n
{
    if n == 0 {
        // add(0,m)=m and add(m,0)=m
        lemma_add_0_r(m);
    } else {
        // IH: add(n-1, m) = add(m, n-1)
        lemma_add_comm((n - 1) as nat, m);

        // Need: add(n, m) = add(m, n)
        // LHS: add(n,m) = add(n-1,m)+1
        // RHS: add(m,n) = add(m,n-1)+1
        lemma_add_succ_r(m, (n - 1) as nat);
    }
}

// A second example where the “obvious” induction variable is wrong:
// prove add(n, m) = add(m, n) by induction on m *also* works, but needs the right lemma.
pub proof fn lemma_add_comm_alt(n: nat, m: nat)
    ensures add(n, m) == add(m, n)
    decreases m
{
    if m == 0 {
        lemma_add_0_r(n);
    } else {
        lemma_add_comm_alt(n, (m - 1) as nat);
        lemma_add_succ_r(n, (m - 1) as nat);
        // add(m,n) unfolds on m, so we also need succ on the other side
        // via commutativity on smaller m:
        lemma_add_comm((m - 1) as nat, n);
    }
}

// ----------------------------
// A derived induction principle: strong induction (implemented as recursion)
// ----------------------------

pub open spec fn sum_to(n: nat) -> nat
    decreases n
{
    if n == 0 { 0 } else { sum_to((n - 1) as nat) + ((n - 1) as nat) }
}

// A small derived principle about this recursion.
pub proof fn lemma_sum_to_succ(n: nat)
    ensures sum_to(n + 1) == sum_to(n) + n
{
    // By unfolding sum_to(n+1): it reduces to sum_to(n) + n.
}

pub proof fn lemma_sum_to_monotone(n: nat)
    ensures sum_to(n) <= sum_to(n + 1)
{
    lemma_sum_to_succ(n);
    assert(sum_to(n + 1) == sum_to(n) + n);
}

// ----------------------------
// An induction principle over user-defined recursion
// ----------------------------

pub proof fn lemma_mul_0_r(n: nat)
    ensures mul(n, 0) == 0
    decreases n
{
    if n == 0 {
    } else {
        lemma_mul_0_r((n - 1) as nat);
        // mul(n,0) = add(0, mul(n-1,0))
        // add(0,x)=x
        lemma_add_0_l(mul((n - 1) as nat, 0));
    }
}

pub fn main() {
    proof {
        lemma_add_0_l(10);
        lemma_add_0_r(10);
        lemma_add_succ_r(4, 7);
        lemma_add_assoc(2, 3, 4);

        lemma_add_comm(5, 8);
        lemma_add_comm_alt(5, 8);

        lemma_sum_to_succ(10);
        lemma_sum_to_monotone(10);

        lemma_mul_0_r(9);
    }
}

} // verus!
