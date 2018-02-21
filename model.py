"""
Generate graph and run model for the sharing behavior on web media
Creator: Eric Araujo
Date: 2017-12-20
"""

import numpy as np
import networkx as nx
import pandas as pd
import math
import json
import sys
from random import random


def generate_graph(weightList=None):
    """
    Inputs: weightList with ((source,target),weight) values
    """
    try:
        edges_f = open('data/connections.csv')
        nodes_f = open('data/states.csv')
    except:
        print("Files absent: connections.csv and states.csv not included in the data folder!")
        sys.exit(0)
    # Initiate graph as digraph (oriented graph)
    graph = nx.DiGraph()
    # Insert nodes
    for line in nodes_f:
        # Read each line and split to get nodes' name and function
        node, func = line.replace(" ", "").strip().split(',')
        # Avoiding include repeated nodes
        if node not in graph.nodes():
            # If node is output
            if node in ['is_pe']:
                graph.add_node(node, attr_dict={'pos': 'output', 'func': func, 'status': {}})
            # If node is internal state
            elif func in ['id', 'alogistic', 'special']:
                graph.add_node(node, attr_dict={'pos': 'inner', 'func': func, 'status': {}})
            # If node is a trait of the participant
            elif func == 'trait':
                graph.add_node(node, attr_dict={'pos': 'trait', 'func': func, 'status': {}})
            # If node is an input
            elif func == 'input':
                graph.add_node(node, attr_dict={'pos': 'input', 'func': func, 'status': {}})
            else:
                print('Node %s does not match the requirements to create graph.', node)
                sys.exit(0)
        else:
            print('<CONFLICT> Node %s already included in the list!', node)
            sys.exit(0)

    outWeightList = []

    # Insert edges
    if weightList is None:
        for line in edges_f:
            source, target, w = line.replace(" ", "").strip().split(',')

            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))
    # In case you have changes in the edges over time.
    else:
        for line in weightList:
            ((source, target), w) = line
            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))

    return graph, outWeightList


def save_graph(graph):
    nx.draw_spring(graph, with_labels = True)
    plt.draw()
    #plt.show()
    plt.savefig('graph_with_labels.png')
    plt.clf()

"""
Input: value to calculate
tau: threshold
sigma: steepness
"""
def alogistic(c, tau, sigma):
    return ((1/(1+math.exp(-sigma*(c-tau))))-(1/(1+math.exp(sigma*tau))))*(1+math.exp(-sigma*tau))
    

"""
Inputs:     message sentiment, message political position, message quality - [msg_s, msg_p,msg_q]
            time of exposure - timesteps
            alogistic_parameters is a dictionary with the tau and sigma for each node that uses alogistic 
            states should be a vector [pp_cons, pp_lib, cs_cons, cs_lib, mood] for the agent to start with
Outputs:    graph with the values for the states
            list of weights used to run the model
            return graph, outWeightList, set_output, alogistic_parameters
"""
def run_message(message=None, traits=None, previous_status_dict=None,
                parameters=None, speed_factor=0.5, delta_t=1,
                timesteps=30, weightList=None):
    # Checking the values for the function
    if message is None or len(message) != 5:
        print('Pass the values of the message correctly to the function!')
        print(message)
        print(message.shape)
        sys.exit()
    if traits is None or len(traits) != 2:
        print('Pass the values of the traits correctly to the function!')
        sys.exit()
    #if previous_status_dict == None:
    #   print 'Starting from zero!'
        
    # Read the json file with the alogistic parameters
    
    if parameters is None:
        try:
            with open('data/parameters.json') as data_file:    
                parameters = json.load(data_file)
        except:
            print('Couldn\'t read the parameters! Check the \'parameters.json\' file!')
            sys.exit()
    elif parameters == 'random':
        parameters = {
                     "srs_sal": [random()*10, random()*20],
                     "arousal": [0.45, random()*20],
                     "attention_1": [2.23, random()*20],
                     "attention_2": [0.23, random()*20],
                     "mood": [5.3, random()*20],
                     "ff_ko": [1.75, random()*20],
                     "ff_ent": [1.43, random()*20],
                     "ff_si": [1.12, random()*20],
                     "ff_is": [2.04, random()*20],
                     "ff_se": [2.45, random()*20],
                     "satisfaction": [2.1, random()*20],
                     "dissatisfaction" : [0.2,random()*20],
                     "prep_like" : [2.8,random()*20],
                     "prep_comm": [3.5,random()*20],
                     "prep_share" : [2.1, random()*20],
                     "mood_speed": random()
                    }
    
    
    graph, outWeightList = generate_graph(weightList)

    rng = np.arange(0.0, timesteps*delta_t, delta_t)
    pos = None
    for t in rng:
        # Initialize the nodes
        if t == 0:
            for node in graph.nodes():
                try:
                    func = graph.nodes[node]['attr_dict']['func']
                    pos = graph.nodes[node]['attr_dict']['pos']
                    #print(node, func, pos)
                except:
                    print('node without func or pos %s at time %i' % (node, t))
                
                # Inputs receive a stable value for all the timesteps
                # message[0] is the time of the message
                if pos == 'input':
                    if node == 'msg_e':
                        graph.nodes[node]['status'] = {0:message[1]}
                    elif node == 'msg_c':
                        graph.nodes[node]['status'] = {0:message[2]}
                    elif node == 'msg_p':
                        graph.nodes[node]['status'] = {0:message[3]}
                    elif node == 'msg_s':
                        graph.nodes[node]['status'] = {0:message[4]}
                    else:
                        print('Node with wrong value:', node)
                        sys.exit()
                # states are the personality traits of the agent
                elif node == 'pe_state':
                    graph.nodes[node]['status'] = {0:traits[0]}
                elif node == 'po_state':
                    graph.nodes[node]['status'] = {0:traits[1]}
                # The other states are set to previous values at the beginning
                else:
                    if previous_status_dict is None:
                        graph.nodes[node]['status'] = {0:0}
                    # Keeping the state of the nodes from previous timestep
                    else:
                        graph.nodes[node]['status'] = {0:previous_status_dict[node]}
            continue

        for node in graph.nodes:
            '''
                For each node (not 0 nodes...):
                    get the neighbors
                    get the function
                    get the weights for the edges
                    calculate the new status value for the node in time t
            '''

            func = graph.nodes[node]['attr_dict']['func']
            pos = graph.nodes[node]['attr_dict']['pos']

            # Get previous state
            try:
                previous_state = graph.nodes[node]['status'][t - delta_t]
            except:
                print(graph.nodes[node]['status'], t, delta_t, node)
                print(graph.nodes[node]['attr_dict']['pos'])

            if node not in ['msg_c', 'msg_s']:
                # If it is identity, the operation is based on the only neighbor.
                if func == 'id' or pos=='input':
                    try:
                        weight = graph.edges[list(graph.predecessors(node))[0], node]['weight']
                        state_pred = graph.nodes[list(graph.predecessors(node))[0]]['status'][t - delta_t]
                        if weight < 0:
                            graph.nodes[node]['status'][t] = previous_state + speed_factor * ((1-abs(weight) * state_pred) - previous_state) * delta_t
                        else:
                            graph.nodes[node]['status'][t] = previous_state + speed_factor * (weight * state_pred - previous_state) * delta_t
                    except:
                        #print('<time ', t, '> node:', list(graph.predecessors(node))[0], '-> ', node, '(id)')
                        print(node, list(graph.predecessors(node)))
                        print(t - delta_t)
                
                elif func=='special' or func=='trait':
                    # po_msg = prod
                    if node == 'po_msg':
                        values_v = []
                        for neig in graph.predecessors(node):
                            neig_w = graph.edges[neig, node]['weight']
                            neig_s = graph.node[neig]['status'][t - delta_t]
                            values_v.append(neig_w*neig_s)
                        c = np.prod(values_v)

                        graph.node[node]['status'][t] = previous_state + speed_factor * (c - previous_state) * delta_t
                    
                    # pe_state = absdiff
                    elif node == 'pe_state':
                        values_v = []
                        for neig in graph.predecessors(node):
                            # Ignore weight if it is != 1
                            neig_w = graph.edges[neig, node]['weight']
                            neig_s = graph.node[neig]['status'][t - delta_t]
                            values_v.append(neig_w*neig_s)

                        c = 2*abs(abs(values_v[0]-values_v[1])-1)-1
                        #print(parameters['pe_state_speed'] )
                        graph.node[node]['status'][t] = previous_state + parameters['pe_state_speed'] * (c - previous_state) * delta_t

                    # pi_msg = ssum
                    elif node == 'pi_msg':
                        tau = 2
                        values_v = []
                        for neig in graph.predecessors(node):
                            # Ignore weight if it is != 1
                            neig_w = graph.edges[neig, node]['weight']
                            neig_s = graph.node[neig]['status'][t - delta_t]
                            values_v.append(neig_w*neig_s)

                        c = np.sum(values_v)/tau
                        graph.node[node]['status'][t] = previous_state + speed_factor * (c - previous_state) * delta_t

                    # po_state = ssum2
                    elif node == 'po_state':
                        tau = 2
                        values_v = []
                        for neig in graph.predecessors(node):

                            neig_w = graph.edges[neig, node]['weight']
                            neig_s = graph.node[neig]['status'][t - delta_t]
                            if neig == 'po_msg':
                                values_v.append(np.power(neig_w*neig_s, 2))
                            elif neig == 'pi_msg':
                                values_v.append(neig_w*neig_s/2)
                            else:
                                values_v.append(neig_w*neig_s)

                        c = np.sum(values_v)/tau
                        graph.node[node]['status'][t] = previous_state + parameters['po_state_speed'] * (c - previous_state) * delta_t
                    else:
                        print('Something is really wrong here!')

            # In case of inputs, copy the previous state again
            else:
                graph.nodes[node]['status'][t] = graph.nodes[node]['status'][t - delta_t]

    # Previous status dictionary to keep track of what was done
    psd = {}
    for node in graph.nodes():
        psd[node] = graph.nodes[node]['status'][t]

    # all these states (apart from mood) should be the same over the simulation
    set_traits = {"pe_state": graph.nodes['pe_state']['status'][t],
                  "po_state": graph.nodes['po_state']['status'][t]
                 }
    return graph, outWeightList, set_traits, parameters, psd


def run_message_sequence(message_seq=None, traits=None, parameters=None, title='0'):
    '''
    Run a sequence of messages for one agent with specific traits and an initial state
    message_seq: array of messages
    traits:
    alogistic_parameters:
    title: Title for graphics to be plotted.
    '''
    timesteps = 20
    delta_t = 1 
    speed_factor = 0.8
    weightList=None
    
    # Initialize empty df
    inputsDF = pd.DataFrame()
    # previous_states_dict
    psd = None

    for message in message_seq:
        if psd is None:
            g, w, set_traits, parameters, psd = run_message(message=message, weightList=weightList, 
                traits=traits, parameters=parameters, speed_factor=speed_factor, delta_t=delta_t, timesteps=timesteps)
        else:
            g, w, set_traits, parameters, psd = run_message(message=message, weightList=weightList,
                traits=list(set_traits.values()), previous_status_dict=psd, parameters=parameters,
                speed_factor=speed_factor, delta_t=delta_t, timesteps=timesteps)

        status_results = {}
        for node in g.nodes():
            status_results[node] = g.node[node]['status']

        inputsDF = inputsDF.append(pd.DataFrame(status_results), ignore_index=True)

    return inputsDF, parameters