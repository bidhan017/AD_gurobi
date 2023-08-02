from pydot import Dot, Edge, Node
import argparse

def create_diagram(path, states, start_state, final_state, transition_dict):
    '''Create diagram of DFA using pydot library and save it in png format

    Parameters:
    path (str): Path to the file where diagram will be saved
    states (set): Set of states in DFA
    start_state (str): Initial state of DFA
    final_state (set): Set of final states in DFA
    transition_dict (dict): Dictionary representing the transitions of DFA
    '''
    graph = Dot(graph_type='digraph', rankdir='LR')
    nodes = {}
    for state in states:
        if state == start_state:
            # color start state with green
            if state in final_state:
                initial_state_node = Node(
                    state,
                    style='filled',
                    peripheries=2,
                    fillcolor='#66cc33')
            else:
                initial_state_node = Node(
                    state, style='filled', fillcolor='#66cc33')
            nodes[state] = initial_state_node
            graph.add_node(initial_state_node)
        else:
            if state in final_state:
                state_node = Node(state, peripheries=2)
            else:
                state_node = Node(state)
            nodes[state] = state_node
            graph.add_node(state_node)
    # adding edges
    for from_state, lookup in transition_dict.items():
        for to_label, to_state in lookup.items():
            graph.add_edge(Edge(
                nodes[from_state],
                nodes[to_state],
                label=to_label
            ))
    if path:
        graph.write_png(path)
    return graph


def create_transition_dict(states, alphabet, t_values):
    '''Create a transition dictionary representing transitions in a DFA.

    Parameters:
        states (set): set of states in the DFA.
        alphabet (set): set of symbols in the DFA's alphabet.
        t_values (list): list of tuples representing transitions, where each tuple contains (current_state, symbol, next_state).

    Returns:
        transition_dict (dict): dictionary representing transitions in the DFA. The keys are states, and the values are dictionaries with symbols as keys and next states as values.
    '''
    transition_dict = {}

    for state in states:
        transition_dict[state] = {}
        for symbol in alphabet:
            transition_dict[state][symbol] = None

    for trans in t_values:
        current_state, symbol, next_state = trans
        transition_dict[current_state][symbol] = next_state

    return transition_dict


def parse_float_list(s):
    try:
        # Split the input string into a list of floats
        float_list = [float(item) for item in s.split(',')]
        return float_list
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid float value in the list")

def make_lists_equal_length(list1, list2):
    if len(list1) == len(list2):
        return list1, list2

    if len(list1) < len(list2):
        list1 += ['2'] * (len(list2) - len(list1))
    else:
        list2 += ['2'] * (len(list1) - len(list2))

    return list1, list2