use vstd::prelude::*;

verus! {

// Basics (Software Foundations, LF/Basics) style snippets in Verus.
// Goal: small, self-contained spec/proof examples that Verus can verify.

pub open spec fn negb(b: bool) -> bool {
    !b
}

pub open spec fn andb(b1: bool, b2: bool) -> bool {
    b1 && b2
}

pub open spec fn orb(b1: bool, b2: bool) -> bool {
    b1 || b2
}

pub open spec fn nandb(b1: bool, b2: bool) -> bool {
    negb(andb(b1, b2))
}

pub open spec fn beq_nat(n: nat, m: nat) -> bool
    decreases n
{
    if n == 0 {
        m == 0
    } else if m == 0 {
        false
    } else {
        beq_nat((n - 1) as nat, (m - 1) as nat)
    }
}

pub open spec fn add(n: nat, m: nat) -> nat
    decreases n
{
    if n == 0 {
        m
    } else {
        add((n - 1) as nat, m) + 1
    }
}

pub open spec fn mul(n: nat, m: nat) -> nat
    decreases n
{
    if n == 0 {
        0
    } else {
        add(mul((n - 1) as nat, m), m)
    }
}

// 1) negb_involutive
pub proof fn ex1_negb_involutive(b: bool)
    ensures negb(negb(b)) == b
{
    if b {
        assert(negb(b) == false);
        assert(negb(negb(b)) == true);
    } else {
        assert(negb(b) == true);
        assert(negb(negb(b)) == false);
    }
}

// 2) andb_commutative
pub proof fn ex2_andb_comm(b1: bool, b2: bool)
    ensures andb(b1, b2) == andb(b2, b1)
{
    match (b1, b2) {
        (true, true) => {
            assert(andb(b1, b2));
            assert(andb(b2, b1));
        }
        (true, false) => {
            assert(!andb(b1, b2));
            assert(!andb(b2, b1));
        }
        (false, true) => {
            assert(!andb(b1, b2));
            assert(!andb(b2, b1));
        }
        (false, false) => {
            assert(!andb(b1, b2));
            assert(!andb(b2, b1));
        }
    }
}

// 3) orb_assoc
pub proof fn ex3_orb_assoc(a: bool, b: bool, c: bool)
    ensures orb(a, orb(b, c)) == orb(orb(a, b), c)
{
    // Pure boolean algebra; case split keeps it solver-friendly.
    match (a, b, c) {
        (false, false, false) => {}
        (false, false, true) => {}
        (false, true, false) => {}
        (false, true, true) => {}
        (true, false, false) => {}
        (true, false, true) => {}
        (true, true, false) => {}
        (true, true, true) => {}
    }
}

// 4) nandb definition sanity
pub proof fn ex4_nandb_def(b1: bool, b2: bool)
    ensures nandb(b1, b2) == !(b1 && b2)
{
    assert(nandb(b1, b2) == negb(andb(b1, b2)));
    assert(negb(andb(b1, b2)) == !(b1 && b2));
}

// 5) beq_nat_refl
pub proof fn ex5_beq_nat_refl(n: nat)
    ensures beq_nat(n, n)
    decreases n
{
    if n == 0 {
        assert(beq_nat(0, 0));
    } else {
        let n1 = (n - 1) as nat;
        ex5_beq_nat_refl(n1);
        assert(n != 0);
        assert(beq_nat(n, n) == beq_nat(n1, n1));
    }
}

// 6) add_0_n
pub proof fn ex6_add_0_n(n: nat)
    ensures add(0, n) == n
{
    assert(add(0, n) == n);
}

// 7) add_n_0
pub proof fn ex7_add_n_0(n: nat)
    ensures add(n, 0) == n
    decreases n
{
    if n == 0 {
        assert(add(0, 0) == 0);
    } else {
        let n1 = (n - 1) as nat;
        ex7_add_n_0(n1);
        assert(add(n, 0) == add(n1, 0) + 1);
        assert(n == n1 + 1);
    }
}

// 8) add_succ_l ("plus" on left successor)
pub proof fn ex8_add_succ_l(n: nat, m: nat)
    ensures add(n + 1, m) == add(n, m) + 1
{
    assert(add(n + 1, m) == add(n, m) + 1);
}

// 9) mul_0_l
pub proof fn ex9_mul_0_l(m: nat)
    ensures mul(0, m) == 0
{
    assert(mul(0, m) == 0);
}

// 10) mul_n_0
pub proof fn ex10_mul_n_0(n: nat)
    ensures mul(n, 0) == 0
    decreases n
{
    if n == 0 {
        assert(mul(0, 0) == 0);
    } else {
        let n1 = (n - 1) as nat;
        ex10_mul_n_0(n1);
        assert(mul(n, 0) == add(mul(n1, 0), 0));
        // add(x, 0) == x (proved above)
        ex7_add_n_0(mul(n1, 0));
    }
}

pub fn main() {
    // A couple of concrete checks (like SF's Compute/Example blocks).
    assert(negb(true) == false);
    assert(andb(true, false) == false);
    assert(orb(false, true) == true);

    // Theorems (proved above). Not strictly necessary to call, but it
    // mirrors "Example/Theorem" blocks in Basics.
    proof {
        ex1_negb_involutive(true);
        ex1_negb_involutive(false);

        ex2_andb_comm(true, true);
        ex2_andb_comm(true, false);
        ex2_andb_comm(false, true);
        ex2_andb_comm(false, false);

        ex3_orb_assoc(false, false, false);
        ex3_orb_assoc(true, false, true);

        ex4_nandb_def(true, true);
        ex4_nandb_def(true, false);

        ex5_beq_nat_refl(0);
        ex5_beq_nat_refl(5);

        ex6_add_0_n(7);
        ex7_add_n_0(7);
        ex8_add_succ_l(3, 4);

        ex9_mul_0_l(9);
        ex10_mul_n_0(6);
    }
}

} // verus!
