use vstd::prelude::*;

verus! {

// Induction (Software Foundations, LF/Induction) style snippets in Verus.
// Focus: proofs by (structural) induction over nat.

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

// A small helper: successor on the left (definitional for our `add`).
pub proof fn lemma_add_succ_l(x: nat, m: nat)
    ensures add(x + 1, m) == add(x, m) + 1
{
    // `x + 1` is always nonzero for nat, so unfolding is safe.
    assert(add(x + 1, m) == add(x, m) + 1);
}

// 1) plus_n_O
pub proof fn ex1_plus_n_O(n: nat)
    ensures add(n, 0) == n
    decreases n
{
    if n == 0 {
        assert(add(0, 0) == 0);
    } else {
        let n1 = (n - 1) as nat;
        ex1_plus_n_O(n1);
        assert(add(n, 0) == add(n1, 0) + 1);
        assert(n == n1 + 1);
    }
}

// 2) plus_n_Sm
pub proof fn ex2_plus_n_Sm(n: nat, m: nat)
    ensures add(n, m + 1) == add(n, m) + 1
    decreases n
{
    if n == 0 {
        assert(add(0, m + 1) == m + 1);
        assert(add(0, m) + 1 == m + 1);
    } else {
        let n1 = (n - 1) as nat;
        ex2_plus_n_Sm(n1, m);
        assert(add(n, m + 1) == add(n1, m + 1) + 1);
        assert(add(n, m) == add(n1, m) + 1);
    }
}

// 3) plus_assoc
pub proof fn ex3_plus_assoc(a: nat, b: nat, c: nat)
    ensures add(add(a, b), c) == add(a, add(b, c))
    decreases a
{
    if a == 0 {
        assert(add(add(0, b), c) == add(b, c));
        assert(add(0, add(b, c)) == add(b, c));
    } else {
        let a1 = (a - 1) as nat;
        ex3_plus_assoc(a1, b, c);

        assert(a == a1 + 1);
        assert(add(a, b) == add(a1, b) + 1);

        // Rewrite LHS using succ-left lemma on (add(a1,b)).
        lemma_add_succ_l(add(a1, b), c);
        assert(add(add(a, b), c) == add(add(a1, b), c) + 1);

        // Rewrite RHS using definitional unfolding at `a = a1 + 1`.
        assert(add(a, add(b, c)) == add(a1, add(b, c)) + 1);

        // Close via IH.
        assert(add(add(a1, b), c) == add(a1, add(b, c)));
    }
}

// 4) plus_comm
pub proof fn ex4_plus_comm(a: nat, b: nat)
    ensures add(a, b) == add(b, a)
    decreases a
{
    if a == 0 {
        assert(add(0, b) == b);
        ex1_plus_n_O(b);
        assert(add(b, 0) == b);
    } else {
        let a1 = (a - 1) as nat;
        ex4_plus_comm(a1, b);

        // LHS: add(a1+1,b) = add(a1,b)+1
        assert(add(a, b) == add(a1, b) + 1);

        // RHS: add(b,a1+1) = add(b,a1)+1 (succ on the right)
        ex2_plus_n_Sm(b, a1);
        assert(add(b, a) == add(b, a1) + 1);

        // Use IH: add(a1,b) = add(b,a1)
        assert(add(a1, b) == add(b, a1));
    }
}

// 5) A rearrangement lemma used frequently in Induction proofs.
//    add(x, add(y, z)) == add(y, add(x, z))
pub proof fn ex5_add_move_left(x: nat, y: nat, z: nat)
    ensures add(x, add(y, z)) == add(y, add(x, z))
{
    // add(x, add(y,z)) = add(add(x,y), z)
    ex3_plus_assoc(x, y, z);
    assert(add(x, add(y, z)) == add(add(x, y), z));

    // swap x and y in add(x,y)
    ex4_plus_comm(x, y);
    assert(add(x, y) == add(y, x));

    // add(add(y,x), z) = add(y, add(x,z))
    ex3_plus_assoc(y, x, z);
    assert(add(add(y, x), z) == add(y, add(x, z)));

    // Now the solver can connect the equalities.
}

// 6) beq_nat_refl
pub proof fn ex6_beq_nat_refl(n: nat)
    ensures beq_nat(n, n)
    decreases n
{
    if n == 0 {
        assert(beq_nat(0, 0));
    } else {
        let n1 = (n - 1) as nat;
        ex6_beq_nat_refl(n1);
        assert(beq_nat(n, n) == beq_nat(n1, n1));
    }
}

// 7) beq_nat symmetry
pub proof fn ex7_beq_nat_sym(n: nat, m: nat)
    ensures beq_nat(n, m) == beq_nat(m, n)
    decreases n
{
    if n == 0 {
        if m == 0 {
            assert(beq_nat(0, 0));
        } else {
            assert(beq_nat(0, m) == false);
            assert(beq_nat(m, 0) == false);
        }
    } else {
        if m == 0 {
            assert(beq_nat(n, 0) == false);
            assert(beq_nat(0, n) == false);
        } else {
            let n1 = (n - 1) as nat;
            let m1 = (m - 1) as nat;
            ex7_beq_nat_sym(n1, m1);
            assert(beq_nat(n, m) == beq_nat(n1, m1));
            assert(beq_nat(m, n) == beq_nat(m1, n1));
        }
    }
}

// 8) beq_nat true implies equality
pub proof fn ex8_beq_nat_true_implies_eq(n: nat, m: nat)
    requires beq_nat(n, m)
    ensures n == m
    decreases n
{
    if n == 0 {
        assert(m == 0);
    } else {
        if m == 0 {
            // beq_nat(n,0) is false when n>0, contradicting the precondition.
            assert(beq_nat(n, 0) == false);
            assert(false);
        } else {
            let n1 = (n - 1) as nat;
            let m1 = (m - 1) as nat;
            assert(beq_nat(n, m) == beq_nat(n1, m1));
            ex8_beq_nat_true_implies_eq(n1, m1);
            assert(n == n1 + 1);
            assert(m == m1 + 1);
        }
    }
}

// 9) mult_n_O
pub proof fn ex9_mult_n_O(n: nat)
    ensures mul(n, 0) == 0
    decreases n
{
    if n == 0 {
        assert(mul(0, 0) == 0);
    } else {
        let n1 = (n - 1) as nat;
        ex9_mult_n_O(n1);
        assert(mul(n, 0) == add(mul(n1, 0), 0));
        ex1_plus_n_O(mul(n1, 0));
    }
}

// 10) left distributivity of mul over add (by induction on n)
pub proof fn ex10_mult_distr_l(n: nat, a: nat, b: nat)
    ensures mul(n, add(a, b)) == add(mul(n, a), mul(n, b))
    decreases n
{
    if n == 0 {
        assert(mul(0, add(a, b)) == 0);
        assert(mul(0, a) == 0);
        assert(mul(0, b) == 0);
        assert(add(mul(0, a), mul(0, b)) == add(0, 0));
        assert(add(0, 0) == 0);
    } else {
        let n1 = (n - 1) as nat;
        ex10_mult_distr_l(n1, a, b);

        // Expand mul at n = n1 + 1.
        assert(mul(n, add(a, b)) == add(mul(n1, add(a, b)), add(a, b)));
        assert(mul(n, a) == add(mul(n1, a), a));
        assert(mul(n, b) == add(mul(n1, b), b));

        // Use IH inside.
        assert(mul(n1, add(a, b)) == add(mul(n1, a), mul(n1, b)));

        // Now rearrange:
        // add(add(xa, xb), add(a,b)) == add(add(xa,a), add(xb,b))
        let xa = mul(n1, a);
        let xb = mul(n1, b);

        // LHS -> add(xa, add(xb, add(a,b)))
        ex3_plus_assoc(xa, xb, add(a, b));
        assert(add(add(xa, xb), add(a, b)) == add(xa, add(xb, add(a, b))));

        // Move xb past a: add(xb, add(a,b)) == add(a, add(xb,b))
        ex5_add_move_left(xb, a, b);
        assert(add(xb, add(a, b)) == add(a, add(xb, b)));
        assert(add(xa, add(xb, add(a, b))) == add(xa, add(a, add(xb, b))));

        // Back to pair form: add(xa, add(a, t)) == add(add(xa,a), t)
        ex3_plus_assoc(xa, a, add(xb, b));
        assert(add(xa, add(a, add(xb, b))) == add(add(xa, a), add(xb, b)));

        // Finish: connect all expansions.
        assert(add(mul(n1, add(a, b)), add(a, b)) == add(add(xa, a), add(xb, b)));
        assert(add(mul(n, a), mul(n, b)) == add(add(xa, a), add(xb, b)));
    }
}

pub fn main() {
    proof {
        ex1_plus_n_O(10);
        ex2_plus_n_Sm(5, 7);
        ex3_plus_assoc(2, 3, 4);
        ex4_plus_comm(6, 9);
        ex6_beq_nat_refl(8);
        ex7_beq_nat_sym(4, 7);
        ex9_mult_n_O(12);
        ex10_mult_distr_l(5, 2, 7);
    }
}

} // verus!
