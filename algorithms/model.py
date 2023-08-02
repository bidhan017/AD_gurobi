
import gurobipy as gp
from gurobipy import GRB
import numpy as np
#from pydot import Dot, Edge, Node
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import itertools
from time import time
from automata.fa.dfa import DFA
from preprocessing import data_preprocess
from utils import create_transition_dict, create_diagram


def train(n, path_train, Dist, eps, eps1, eps2):
    '''Train a DFA using Gurobi optimization.

    Parameters: 
    n (int) : Number of states in the DFA
    path_train (str): path of training dataset
    Dist (function): Distance function to use for training
    model_path (str): path to store the model file
    diagram_path (str): path to store model diagram
    eps, eps1, eps2 (float): Regularization parameter for the constraints in Objective function 

    Returns:
    dfa1: Trained DFA model
    '''
    #Pre-process the dataset and return required variables
    alphabet, Pref_S, Lst, FL, dist = data_preprocess(path_train, Dist)
    states = {str(f'q{i}') for i in range(n)}
    start_state = 'q0'

    #Initialize gurobi environment
    env=gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    t0 = time()
    
    # Creating a new Gurobi model
    m = gp.Model("DFA_LST", env=env)

    #DECISION VARIABLES
    delta = m.addVars(states, alphabet, states, vtype=gp.GRB.BINARY, name='delta')
    x = m.addVars(Pref_S, states, vtype=gp.GRB.BINARY, name='x')
    f = m.addVars(states, vtype=gp.GRB.BINARY, name='f')
    alpha = m.addVars(len(Lst), states, vtype=gp.GRB.BINARY, name= 'alpha')
    beta = m.addVars(len(Lst), states, vtype=gp.GRB.BINARY, name= 'beta')

    #OBJECTIVE FUNCTION
    print(f'eps:{eps}, eps1:{eps1}, eps2:{eps2}')
    lambda_nn = len(FL)*(len(FL)-1)*np.max(dist)
    #lambda_na = len(FL)*len(FL)*np.max(dist)
    
    m.setObjective(sum(beta[i,s0]*beta[k,s1]*dist[i,k]*(eps/lambda_nn) for i,_ in enumerate(FL) for s0 in states for k,_ in enumerate(FL) for s1 in states if dist[i,k] != 0) \
                    #+ sum(beta[i,s0]*alpha[k,s1]*(np.max(dist)-dist[i,k])*(epsilon3/lambda_na) for i,_ in enumerate(FL) for s0 in states for k,_ in enumerate(FL) for s1 in states if (np.max(dist)-dist[i,k]) != 0) \
                    + sum(alpha[k,s1]*(eps1/len(FL)) for k,_ in enumerate(FL) for s1 in states) \
                    + sum(delta[s0,symbol,s1]*eps2 for s0 in states for symbol in alphabet for s1 in states if s0 != s1), \
                          gp.GRB.MINIMIZE)
    
    #AUTOMATA CONSTRAINTS
    #Constraint1 
    for s0 in states:
        for symbol in alphabet:
            m.addConstr(sum(delta[s0,symbol,s1] for s1 in states)==1, name=f'delta[{s0},{symbol}]')
    
    #Constraint2 
    for Pref in Pref_S:
        m.addConstr(sum(x[Pref,s1] for s1 in states)==1, name=f'x[{Pref}]')

    #Constraint3 
    m.addConstr(x['',start_state]==1, name='initial_state')

    #Constraint4 
    for s0, Pref, symbol, s1 in itertools.product(states, Pref_S, alphabet, states):
        if (Pref + ',' + symbol) in Pref_S:
            m.addConstr(x[Pref, s0] + delta[s0, symbol, s1] - 1 <= x[Pref + ',' + symbol, s1], name=f'transition[{s0},{Pref},{symbol},{s1}]')
        if Pref == '' and symbol in Pref_S:
            m.addConstr(x[Pref, s0] + delta[s0, symbol, s1] - 1 <= x[symbol, s1], name=f'transition[{s0},{Pref},{symbol},{s1}]')

    #BOUND CONSTRAINTS (for alpha and beta variables)

    for i, word in enumerate(Lst):
        for s1 in states:
            m.addConstr(alpha[i, s1] >= x[word,s1] + f[s1] -1, name=f'bound_1[{s1},{i}]')        

    for i, word in enumerate(Lst):
        for s1 in states:
            m.addConstr(alpha[i, s1] <= x[word,s1], name=f'bound_2[{s1},{i}]')

    for i, word in enumerate(Lst):
        for s1 in states:
            m.addConstr(alpha[i, s1] <= f[s1], name=f'bound_3[{s1},{i}]')
    
    #not valid MILP constraint
    '''
    for i, word in enumerate(Lst):
        for s1 in states:
            m.addConstr(x[word, s1] * (1-f[s1]) == beta[i, s1], name=f'bound_4[{s1},{i}]')
    '''
    for i, word in enumerate(Lst):
        for s1 in states:
            m.addConstr(beta[i, s1] >= x[word,s1] + (1-f[s1]) -1, name=f'bound_5[{s1},{i}]')

    for i, word in enumerate(Lst):
        for s1 in states:
            m.addConstr(beta[i, s1] <= x[word,s1], name=f'bound_6[{s1},{i}]')

    for i, word in enumerate(Lst):
        for s1 in states:
            m.addConstr(beta[i, s1] <= (1-f[s1]), name=f'bound_7[{s1},{i}]')

    #optimize the model
    m.optimize()
    
    t1 = time()
    print("Run time", (t1-t0), "seconds")

    #write the model
    m.write(f'reports/model_WP_LST_{n}.lp')
    
    if m.status == 1:
        status = 'LOADED'
        print(f'DFAmodel_{n}: {status}')
            
    elif m.status == 2:
        status='OPTIMAL'
        print(f'DFAmodel_{n}: {status}')
        
        transitions = m.getAttr('X', delta)
        t_values = [(s0,a,s1) for s0 in states for s1 in states for a in alphabet if round(transitions[s0, a, s1],0) == 1]
        #for t in t_values:
            #print(t)
        f_s = m.getAttr('X', f)
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

        create_diagram(f'reports/diagram_WP_LST_{n}.png', states, start_state,final_state, transition_dict)
        return dfa1        
    
    elif m.status == 3:
        status = 'INFEASIBLE'
        print(f'DFAmodel_{n}: {status} ')

    else:
        print('status unknown, DEBUG!!')


def test(path_test, correct_label, dfa1):
    '''Test the trained DFA on a test dataset and evaluate its performance

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