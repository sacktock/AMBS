
from ambs.model_checking.dfa import Atom
from ambs.model_checking.dfa import Truth
from ambs.model_checking.dfa import And
from ambs.model_checking.dfa import Or
from ambs.model_checking.dfa import Neg
from ambs.model_checking.dfa import Implies
from ambs.model_checking.dfa import DFA
from ambs.model_checking.dfa import Cost_Function

automaton_states = [0,1,2,3]
initial_state = 0
accepting_states = [1,2]

automaton_cf = DFA(automaton_states, initial_state, accepting_states)

"""define the edges"""

automaton_cf.add_edge(0, 1, Atom('death'))
automaton_cf.add_edge(0, 2, Or(Atom('out-of-oxygen'), Atom('early-surface')))

automaton_cf.add_edge(1, 0, Truth())

automaton_cf.add_edge(2, 3, Truth())

automaton_cf.add_edge(3, 0, Atom('death'))

"""define the cost function"""
cost_function = Cost_Function(automaton_cf)

automaton = DFA(automaton_states, initial_state, accepting_states)

"""define the edges"""

automaton.add_edge(0, 1, Atom('death'))
automaton.add_edge(0, 2, Or(Atom('out-of-oxygen'), Atom('early-surface')))

automaton.add_edge(1, 0, Truth())

automaton.add_edge(2, 3, Truth())

automaton.add_edge(3, 0, Atom('death'))
