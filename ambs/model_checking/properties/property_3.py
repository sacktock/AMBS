from ambs.model_checking.dfa import Atom
from ambs.model_checking.dfa import Truth
from ambs.model_checking.dfa import And
from ambs.model_checking.dfa import Or
from ambs.model_checking.dfa import Neg
from ambs.model_checking.dfa import Implies
from ambs.model_checking.dfa import DFA
from ambs.model_checking.dfa import Cost_Function

automaton_states = [s for s in range(31)]
initial_state = 0
accepting_states = [30]

automaton_cf = DFA(automaton_states, initial_state, accepting_states)

"""define the edges"""
automaton_cf.add_edge(0, 1, And(Atom('diver'), Neg(Atom('surface'))))

automaton_cf.add_edge(1, 0, Atom('surface'))
automaton_cf.add_edge(1, 2, Neg(Atom('surface')))

automaton_cf.add_edge(2, 0, Atom('surface'))
automaton_cf.add_edge(2, 3, Neg(Atom('surface')))

automaton_cf.add_edge(3, 0, Atom('surface'))
automaton_cf.add_edge(3, 4, Neg(Atom('surface')))

automaton_cf.add_edge(4, 0, Atom('surface'))
automaton_cf.add_edge(4, 5, Neg(Atom('surface')))

automaton_cf.add_edge(5, 0, Atom('surface'))
automaton_cf.add_edge(5, 6, Neg(Atom('surface')))

automaton_cf.add_edge(6, 0, Atom('surface'))
automaton_cf.add_edge(6, 7, Neg(Atom('surface')))

automaton_cf.add_edge(7, 0, Atom('surface'))
automaton_cf.add_edge(7, 8, Neg(Atom('surface')))

automaton_cf.add_edge(8, 0, Atom('surface'))
automaton_cf.add_edge(8, 9, Neg(Atom('surface')))

automaton_cf.add_edge(9, 0, Atom('surface'))
automaton_cf.add_edge(9, 10, Neg(Atom('surface')))

automaton_cf.add_edge(10, 0, Atom('surface'))
automaton_cf.add_edge(10, 11, Neg(Atom('surface')))

automaton_cf.add_edge(11, 0, Atom('surface'))
automaton_cf.add_edge(11, 12, Neg(Atom('surface')))

automaton_cf.add_edge(12, 0, Atom('surface'))
automaton_cf.add_edge(12, 13, Neg(Atom('surface')))

automaton_cf.add_edge(13, 0, Atom('surface'))
automaton_cf.add_edge(13, 14, Neg(Atom('surface')))

automaton_cf.add_edge(14, 0, Atom('surface'))
automaton_cf.add_edge(14, 15, Neg(Atom('surface')))

automaton_cf.add_edge(15, 0, Atom('surface'))
automaton_cf.add_edge(15, 16, Neg(Atom('surface')))

automaton_cf.add_edge(16, 0, Atom('surface'))
automaton_cf.add_edge(16, 17, Neg(Atom('surface')))

automaton_cf.add_edge(17, 0, Atom('surface'))
automaton_cf.add_edge(17, 18, Neg(Atom('surface')))

automaton_cf.add_edge(18, 0, Atom('surface'))
automaton_cf.add_edge(18, 19, Neg(Atom('surface')))

automaton_cf.add_edge(19, 0, Atom('surface'))
automaton_cf.add_edge(19, 20, Neg(Atom('surface')))

automaton_cf.add_edge(20, 0, Atom('surface'))
automaton_cf.add_edge(20, 21, Neg(Atom('surface')))

automaton_cf.add_edge(21, 0, Atom('surface'))
automaton_cf.add_edge(21, 22, Neg(Atom('surface')))

automaton_cf.add_edge(22, 0, Atom('surface'))
automaton_cf.add_edge(22, 23, Neg(Atom('surface')))

automaton_cf.add_edge(23, 0, Atom('surface'))
automaton_cf.add_edge(23, 24, Neg(Atom('surface')))

automaton_cf.add_edge(24, 0, Atom('surface'))
automaton_cf.add_edge(24, 25, Neg(Atom('surface')))

automaton_cf.add_edge(25, 0, Atom('surface'))
automaton_cf.add_edge(25, 26, Neg(Atom('surface')))

automaton_cf.add_edge(26, 0, Atom('surface'))
automaton_cf.add_edge(26, 27, Neg(Atom('surface')))

automaton_cf.add_edge(27, 0, Atom('surface'))
automaton_cf.add_edge(27, 28, Neg(Atom('surface')))

automaton_cf.add_edge(28, 0, Atom('surface'))
automaton_cf.add_edge(28, 29, Neg(Atom('surface')))

automaton_cf.add_edge(29, 0, Atom('surface'))
automaton_cf.add_edge(29, 30, Neg(Atom('surface')))

automaton_cf.add_edge(30, 0, Atom('surface'))

"""define the cost function"""
cost_function = Cost_Function(automaton_cf)

automaton = DFA(automaton_states, initial_state, accepting_states)

"""define the edges"""
automaton.add_edge(0, 1, And(Atom('diver'), Neg(Atom('surface'))))

automaton.add_edge(1, 0, Atom('surface'))
automaton.add_edge(1, 2, Neg(Atom('surface')))

automaton.add_edge(2, 0, Atom('surface'))
automaton.add_edge(2, 3, Neg(Atom('surface')))

automaton.add_edge(3, 0, Atom('surface'))
automaton.add_edge(3, 4, Neg(Atom('surface')))

automaton.add_edge(4, 0, Atom('surface'))
automaton.add_edge(4, 5, Neg(Atom('surface')))

automaton.add_edge(5, 0, Atom('surface'))
automaton.add_edge(5, 6, Neg(Atom('surface')))

automaton.add_edge(6, 0, Atom('surface'))
automaton.add_edge(6, 7, Neg(Atom('surface')))

automaton.add_edge(7, 0, Atom('surface'))
automaton.add_edge(7, 8, Neg(Atom('surface')))

automaton.add_edge(8, 0, Atom('surface'))
automaton.add_edge(8, 9, Neg(Atom('surface')))

automaton.add_edge(9, 0, Atom('surface'))
automaton.add_edge(9, 10, Neg(Atom('surface')))

automaton.add_edge(10, 0, Atom('surface'))
automaton.add_edge(10, 11, Neg(Atom('surface')))

automaton.add_edge(11, 0, Atom('surface'))
automaton.add_edge(11, 12, Neg(Atom('surface')))

automaton.add_edge(12, 0, Atom('surface'))
automaton.add_edge(12, 13, Neg(Atom('surface')))

automaton.add_edge(13, 0, Atom('surface'))
automaton.add_edge(13, 14, Neg(Atom('surface')))

automaton.add_edge(14, 0, Atom('surface'))
automaton.add_edge(14, 15, Neg(Atom('surface')))

automaton.add_edge(15, 0, Atom('surface'))
automaton.add_edge(15, 16, Neg(Atom('surface')))

automaton.add_edge(16, 0, Atom('surface'))
automaton.add_edge(16, 17, Neg(Atom('surface')))

automaton.add_edge(17, 0, Atom('surface'))
automaton.add_edge(17, 18, Neg(Atom('surface')))

automaton.add_edge(18, 0, Atom('surface'))
automaton.add_edge(18, 19, Neg(Atom('surface')))

automaton.add_edge(19, 0, Atom('surface'))
automaton.add_edge(19, 20, Neg(Atom('surface')))

automaton.add_edge(20, 0, Atom('surface'))
automaton.add_edge(20, 21, Neg(Atom('surface')))

automaton.add_edge(21, 0, Atom('surface'))
automaton.add_edge(21, 22, Neg(Atom('surface')))

automaton.add_edge(22, 0, Atom('surface'))
automaton.add_edge(22, 23, Neg(Atom('surface')))

automaton.add_edge(23, 0, Atom('surface'))
automaton.add_edge(23, 24, Neg(Atom('surface')))

automaton.add_edge(24, 0, Atom('surface'))
automaton.add_edge(24, 25, Neg(Atom('surface')))

automaton.add_edge(25, 0, Atom('surface'))
automaton.add_edge(25, 26, Neg(Atom('surface')))

automaton.add_edge(26, 0, Atom('surface'))
automaton.add_edge(26, 27, Neg(Atom('surface')))

automaton.add_edge(27, 0, Atom('surface'))
automaton.add_edge(27, 28, Neg(Atom('surface')))

automaton.add_edge(28, 0, Atom('surface'))
automaton.add_edge(28, 29, Neg(Atom('surface')))

automaton.add_edge(29, 0, Atom('surface'))
automaton.add_edge(29, 30, Neg(Atom('surface')))

automaton.add_edge(30, 0, Truth())

