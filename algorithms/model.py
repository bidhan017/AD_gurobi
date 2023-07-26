
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from pydot import Dot, Edge, Node
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import itertools
from time import time
from automata.fa.dfa import DFA
from preprocessing import data_preprocess


def train(n, path_train, Dist, model_path, diagram_path, eps, eps1, eps2):
    '''
    Train a DFA using Gurobi optimization.

    Parameters: 
    n (int) : Number of states in the DFA
    path_train (str): path of training dataset
    Dist (function): Distance function to use for training
    eps, eps1, eps2 (float): Regularization parameter for the constraints in Objective function 

    Returns:
    dfa1: Trained DFA model
    '''

    alphabet, Pref_S, Lst, FL, dist = data_preprocess(path_train, Dist)
    states = {str(f'q{i}') for i in range(n)}
    start_state = 'q0'

    env=gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    t0 = time()
    
    # Creating a new model
    mwplst = gp.Model("DFA_LST", env=env)

    #DECISION VARIABLES
    delta = mwplst.addVars(states, alphabet, states, vtype=gp.GRB.BINARY, name='delta')
    x = mwplst.addVars(Pref_S, states, vtype=gp.GRB.BINARY, name='x')
    f = mwplst.addVars(states, vtype=gp.GRB.BINARY, name='f')
    alpha = mwplst.addVars(len(Lst), states, vtype=gp.GRB.BINARY, name= 'alpha')
    beta = mwplst.addVars(len(Lst), states, vtype=gp.GRB.BINARY, name= 'beta')

    #OBJECTIVE FUNCTION
    print(f'eps:{eps}, eps1:{eps1}, eps2:{eps2}')
    lambda_nn = len(FL)*(len(FL)-1)*np.max(dist)
    #lambda_na = len(FL)*len(FL)*np.max(dist)
    
    mwplst.setObjective(sum(beta[i,state1]*beta[k,state2]*dist[i,k]*(eps/lambda_nn) for i,_ in enumerate(FL) for state1 in states for k,_ in enumerate(FL) for state2 in states if dist[i,k] != 0) \
                    #+ sum(beta[i,state1]*alpha[k,state2]*(np.max(dist)-dist[i,k])*(epsilon3/lambda_na) for i,_ in enumerate(FL) for state1 in states for k,_ in enumerate(FL) for state2 in states if (np.max(dist)-dist[i,k]) != 0) \
                    + sum(alpha[k,state2]*(eps1/len(FL)) for k,_ in enumerate(FL) for state2 in states) \
                    + sum(delta[state1,symbol,state2]*eps2 for state1 in states for symbol in alphabet for state2 in states if state1 != state2), \
                          gp.GRB.MINIMIZE)
    
    #AUTOMATA CONSTRAINTS
    #Constraint1
    for state0 in states:
        for symbol in alphabet:
            mwplst.addConstr(sum(delta[state0,symbol,state1] for state1 in states)==1, name=f'delta[{state0},{symbol}]')

    #Constraint2
    for word in Pref_S:
        mwplst.addConstr(sum(x[word,state1] for state1 in states)==1, name=f'x[{word}]')

    #Constraint3
    mwplst.addConstr(x['',start_state]==1, name='initial_state')

    #Constraint4 
    for state0, word, symbol, state1 in itertools.product(states, Pref_S, alphabet, states):
        if (word + ',' + symbol) in Pref_S:
            mwplst.addConstr(x[word, state0] + delta[state0, symbol, state1] - 1 <= x[word + ',' + symbol, state1], name=f'transition[{state0},{word},{symbol},{state1}]')
        if word == '' and symbol in Pref_S:
            mwplst.addConstr(x[word, state0] + delta[state0, symbol, state1] - 1 <= x[symbol, state1], name=f'transition[{state0},{word},{symbol},{state1}]')

    #BOUND CONSTRAINTS
    for i, word in enumerate(Lst):
        for state1 in states:
            mwplst.addConstr(alpha[i, state1] >= x[word,state1] + f[state1] -1, name=f'bound_1[{state1},{i}]')        

    for i, word in enumerate(Lst):
        for state1 in states:
            mwplst.addConstr(alpha[i, state1] <= x[word,state1], name=f'bound_2[{state1},{i}]')

    for i, word in enumerate(Lst):
        for state1 in states:
            mwplst.addConstr(alpha[i, state1] <= f[state1], name=f'bound_3[{state1},{i}]')
    
    #not valid MILP constraint
    '''
    for i, word in enumerate(Lst):
        for state1 in states:
            mwplst.addConstr(x[word, state1] * (1-f[state1]) == beta[i, state1], name=f'bound_4[{state1},{i}]')
    '''
    for i, word in enumerate(Lst):
        for state1 in states:
            mwplst.addConstr(beta[i, state1] >= x[word,state1] + (1-f[state1]) -1, name=f'bound_5[{state1},{i}]')

    for i, word in enumerate(Lst):
        for state1 in states:
            mwplst.addConstr(beta[i, state1] <= x[word,state1], name=f'bound_6[{state1},{i}]')

    for i, word in enumerate(Lst):
        for state1 in states:
            mwplst.addConstr(beta[i, state1] <= (1-f[state1]), name=f'bound_7[{state1},{i}]')

    #optimize the model
    mwplst.optimize()
    
    t1 = time()
    print("Run time", (t1-t0), "seconds")

    #write the model
    #mwplst.write(rf'C:\Users\bchan\Desktop\TUD\Thesis\model_WP_LST_{n}.lp')
    mwplst.write(model_path)
    
    if mwplst.status == 1:
        status = 'LOADED'
        print(f'DFAmodel_{n} LOADED')
        #dfa1 = DFA(states=states,input_symbols=alphabet, transitions= transition_dict, initial_state= start_state, final_states={'q0'})
    
    elif mwplst.status == 2:
        print(f'DFAmodel_{n} OPTIMAL')
        status='OPTIMAL'
        transitions = mwplst.getAttr('X', delta)
        t_values = [(s1,a,s2) for s1 in states for s2 in states for a in alphabet if round(transitions[s1, a, s2],0) == 1]
        #for t in t_values:
            #print(t)
        f_s = mwplst.getAttr('X', f)
        final_state = {s1 for s1 in states if round(f_s[s1],0) == 1}

        transition_dict = create_transition_dict(states, alphabet, t_values)
        #print(transition_dict)
        dfa1 = DFA(states=states, input_symbols=alphabet, transitions= transition_dict, initial_state= start_state, final_states=final_state)
        accepted = 0
        rejected = 0
        for w in FL:
            #print(w)
            if dfa1.accepts_input(w):
                #print(f'{w}:accepted')
                accepted += 1             
            else:
                #print(f'{w}:rejected')
                rejected += 1        
        print(f'Accepted in Training:{accepted}')
        print(f'Rejected in Training:{rejected}')

        create_diagram(diagram_path, states, start_state,final_state, transition_dict)
        #create_diagram(rf'C:\Users\bchan\Desktop\TUD\Thesis\diagram_WP_LST_{n}.png', states, start_state,final_state, transition_dict)        
        return dfa1        
    
    elif mwplst.status == 3:
        status = 'INFEASIBLE'
        print(f'DFAmodel_{n} INFEASIBLE')
    else:
        print('status unknown, DEBUG!!')    
 
    return status


def create_transition_dict(states, alphabet, t_values):
    transition_dict = {}

    for state in states:
        transition_dict[state] = {}
        for symbol in alphabet:
            transition_dict[state][symbol] = None

    for trans in t_values:
        current_state, symbol, next_state = trans
        transition_dict[current_state][symbol] = next_state

    return transition_dict


def create_diagram(path, states, start_state, final_state, transition_dict):
    '''
    Create diagram of DFA using pydot library and save it in png format

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


def test(path_test, correct_label, dfa1):
    '''
    Test the trained DFA on a test dataset and evaluate its performance

    Parameters:
    path_test (str): Path to test dataset file
    correct_label (int): correct label for the test dataset
    dfa1 (DFA): trained DFA model

    Prints:
    Number of accepted and rejected inputs in the test dataset
    Accuracy and F1_score of DFA on test dataset
    '''

    with open(path_test, "r") as my_file:
        lines = [line.strip() for line in my_file.readlines()]        
    
    #Lst1, FL1 for test dataset is same as Lst and FL for train dataset
    Lst1, FL1, G = [], [], []

    for line in lines:
        Lst_line,g = tuple(line.rstrip().split(";"))
        Lst1.append(Lst_line)
        FL1.append(Lst_line.split(','))
        #change here a/c to true label
        if int(g)==correct_label:
            G.append(0)
        else:
            G.append(1)

    accepted = 0
    rejected = 0
    Predicted_labels=[]
    for w in FL1:
        #print(w)
        if dfa1.accepts_input(w):
            #print(f'{w}:accepted')
            Predicted_labels.append(1)
            accepted += 1             
        else:
            #print(f'{w}:rejected')
            Predicted_labels.append(0)
            rejected += 1
            
    print(f'Accepted in Testing:{accepted}')
    print(f'Rejected in Testing:{rejected}')    
    #print(f'Predicted_labels:{Predicted_labels}')
    #print(f'True_labels:{G}')

    accuracy = accuracy_score(G, Predicted_labels)
    print(f'Accuracy:{round(accuracy,2)}')
    #f1score=[]
    f1 = f1_score(G, Predicted_labels, average='binary', pos_label=1)
    #f1score.append(f1)
    print(f'F1_score:{round(f1,2)}\n')
    #print(f'F1_score_list:{f1score}\n')