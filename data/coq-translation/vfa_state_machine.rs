use vstd::prelude::*;

verus! {

// ============================================================================
// VFA: State Machine (Supporting VFA)
// ============================================================================

pub struct State { pub value: nat }
pub struct Transition { pub from: nat, pub to: nat, pub label: nat }
pub struct StateMachine { pub states: nat, pub transitions: Seq<Transition>, pub initial: nat, pub accepting: Seq<nat> }

pub open spec fn valid_transition(sm: StateMachine, t: Transition) -> bool { t.from < sm.states && t.to < sm.states }
pub open spec fn valid_sm(sm: StateMachine) -> bool { sm.initial < sm.states && forall|i: nat| #![auto] i < sm.transitions.len() ==> valid_transition(sm, sm.transitions[i as int]) }

pub open spec fn can_transition(sm: StateMachine, from: nat, to: nat) -> bool {
    exists|i: nat| #![auto] i < sm.transitions.len() && sm.transitions[i as int].from == from && sm.transitions[i as int].to == to
}

pub open spec fn is_accepting(sm: StateMachine, s: nat) -> bool decreases sm.accepting.len() {
    sm.accepting.len() > 0 && (sm.accepting[0] == s || is_accepting(StateMachine { accepting: sm.accepting.skip(1), ..sm }, s))
}

pub proof fn example_sm() {
    let sm = StateMachine { states: 3, transitions: seq![Transition { from: 0, to: 1, label: 0 }], initial: 0, accepting: seq![2nat] };
    assert(valid_sm(sm));
}

pub proof fn state_machine_verify() { example_sm(); }
pub fn main() { proof { state_machine_verify(); } }

} // verus!
