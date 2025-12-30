use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: Red-Black Tree Definition (vfa-current/Redblack)
// Red-black tree structure and colors
// ============================================================================

// ----------------------------------------------------------------------------
// Color Type
// ----------------------------------------------------------------------------

#[derive(PartialEq, Eq)]
pub enum Color {
    Red,
    Black,
}

// ----------------------------------------------------------------------------
// Red-Black Tree Type
// ----------------------------------------------------------------------------

pub enum RBTree {
    E,  // Empty (implicitly black)
    T {
        color: Color,
        left: Box<RBTree>,
        key: int,
        value: nat,
        right: Box<RBTree>,
    },
}

// ----------------------------------------------------------------------------
// Basic Operations
// ----------------------------------------------------------------------------

// Empty tree
pub open spec fn empty_rb() -> RBTree {
    RBTree::E
}

// Get color of tree
pub open spec fn get_color(t: RBTree) -> Color {
    match t {
        RBTree::E => Color::Black,
        RBTree::T { color, .. } => color,
    }
}

// Make tree black
pub open spec fn make_black(t: RBTree) -> RBTree {
    match t {
        RBTree::E => RBTree::E,
        RBTree::T { color: _, left, key, value, right } =>
            RBTree::T { color: Color::Black, left, key, value, right },
    }
}

// ----------------------------------------------------------------------------
// Lookup
// ----------------------------------------------------------------------------

pub open spec fn rb_lookup(d: nat, k: int, t: RBTree) -> nat
    decreases t
{
    match t {
        RBTree::E => d,
        RBTree::T { color: _, left, key, value, right } =>
            if k < key {
                rb_lookup(d, k, *left)
            } else if k > key {
                rb_lookup(d, k, *right)
            } else {
                value
            }
    }
}

// ----------------------------------------------------------------------------
// Tree Properties
// ----------------------------------------------------------------------------

// Tree size
pub open spec fn rb_size(t: RBTree) -> nat
    decreases t
{
    match t {
        RBTree::E => 0,
        RBTree::T { color: _, left, key: _, value: _, right } =>
            1 + rb_size(*left) + rb_size(*right),
    }
}

// Tree height
pub open spec fn rb_height(t: RBTree) -> nat
    decreases t
{
    match t {
        RBTree::E => 0,
        RBTree::T { color: _, left, key: _, value: _, right } => {
            let lh = rb_height(*left);
            let rh = rb_height(*right);
            1 + if lh > rh { lh } else { rh }
        }
    }
}

// Black height (number of black nodes on any path)
pub open spec fn black_height(t: RBTree) -> nat
    decreases t
{
    match t {
        RBTree::E => 0,
        RBTree::T { color, left, key: _, value: _, right: _ } => {
            let lbh = black_height(*left);
            let inc = if color == Color::Black { 1nat } else { 0nat };
            inc + lbh
        }
    }
}

// ----------------------------------------------------------------------------
// Red-Black Invariants (Simplified)
// ----------------------------------------------------------------------------

// No red node has a red child
pub open spec fn no_red_red(t: RBTree) -> bool
    decreases t
{
    match t {
        RBTree::E => true,
        RBTree::T { color, left, key: _, value: _, right } => {
            let left_ok = no_red_red(*left);
            let right_ok = no_red_red(*right);
            let this_ok = if color == Color::Red {
                get_color(*left) == Color::Black && get_color(*right) == Color::Black
            } else {
                true
            };
            this_ok && left_ok && right_ok
        }
    }
}

// Root is black
pub open spec fn root_black(t: RBTree) -> bool {
    get_color(t) == Color::Black
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

pub proof fn example_empty()
{
    assert(get_color(empty_rb()) == Color::Black);
    assert(root_black(empty_rb()));
    assert(no_red_red(empty_rb()));
}

pub proof fn example_singleton_black()
{
    let t = RBTree::T {
        color: Color::Black,
        left: Box::new(RBTree::E),
        key: 5,
        value: 50,
        right: Box::new(RBTree::E),
    };
    assert(root_black(t));
    reveal_with_fuel(no_red_red, 2);
    assert(no_red_red(t));
}

pub proof fn example_lookup()
{
    reveal_with_fuel(rb_lookup, 3);
    let t = RBTree::T {
        color: Color::Black,
        left: Box::new(RBTree::E),
        key: 5int,
        value: 50,
        right: Box::new(RBTree::E),
    };
    assert(rb_lookup(0, 5int, t) == 50);
    assert(rb_lookup(0, 3int, t) == 0);
}

// ============================================================================
// MAIN
// ============================================================================

pub proof fn redblack_def_verify()
{
    example_empty();
    example_singleton_black();
    example_lookup();

    // Test properties
    let t = empty_rb();
    assert(rb_size(t) == 0);
    assert(rb_height(t) == 0);
    assert(black_height(t) == 0);
}

pub fn main() {
    proof {
        redblack_def_verify();
    }
}

} // verus!
