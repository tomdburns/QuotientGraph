"""
My Attempt at writing a code to generate quotient graphs from a cif

~~~~~~~~~~~~~
VERSION 0.1.3 - Incomplete
~~~~~~~~~~~~~

Known Issues:
> Does not effectively deal with materials without the loops
> Does not effectively deal with materials that have one loop but no others
  > see UBACOR and ZURQOS
> Cannot generate a quotient graph for TULLIW
> Results for UBACOR are also odd

Changelog - v0.1.1
> Added a loop to simplify function that generates the list of node tags prior
  running the tag generation
> Modified tag list to grab tags from new array containing recently generated
  tags that include connectivity information
> Modified all of the tagging in simplify() to allow for more specific classification
  of the nodes based on their connectivities. This ultimately eliminated the issue
  of incorrect connectivity when identical nodes had differing connectivities
> Added a loop that runs graph.py to main() if the structure's .graph file does not
  exist.

Changelog - v0.1.2
> Located source of the clipping bug in clip_graph() function
> Fixed clipping bug - but generated a new bug that removes one of the rings
  from the linker
> Modified remove_redundant_chunks function to favour redundant chunks with
  largest adjacency matrices over smaller matrices. This will hopefully
  solve the 3 ring issue described in "known issues section"
> Incorrect quotient graph bug fixed
> All bug fixes causing known issues with multi-ringed linkers fixed
> Added min and max loop sizes as environment variables

Changelog - v0.1.3
>
"""


VERSION = (0, 1, 2)
VER = '%i.%i.%i' % VERSION


import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import itertools as it
from subprocess import Popen, PIPE
from graph import Cif, Node, Edge, Graph, dist

#infile = 'ZnBDC.graph'
#infile = 'ZnBDC_OH.graph'
#infile = 'ZURQOS_clean.graph' # problematic
#infile = 'ANUGOG_clean.graph'
#infile = 'ADEGIA_clean.graph'
#infile = 'ZnBDC_Super.graph'
#infile = 'ZnBDC_OH_Super.graph'
infile = sys.argv[1]
if '.cif' in infile:
    infile = infile.split('.cif')[0] + '.graph'
ciffile = infile.split('.graph')[0] + '.cif'

tol = 0.0001
TERMS = []
TNODES = []

# Loop Sizes
MAX, MIN = 8, 2


def atom_type(tag):
    """Identifies the element type"""
    if len(tag) > 3:
        try:
            float(tag[-3:])
            return tag[:-3]
        except ValueError:
            pass
    if len(tag) > 2:
        try:
            float(tag[-2:])
            return tag[:-2]
        except ValueError:
            pass
    return tag[:-1]


def print_matrix(matrix, tag=None):
    """Prints a matrix to a txt file"""
    #return None
    if tag is None:
        out = open('matrix.txt', 'w')
    else:
        out = open('%s_matrix.txt' % tag, 'w')
    for row in matrix:
        out.write(','.join(['%i' % i for i in row]) + '\n')
    out.flush()
    out.close()


def locate_fragment(graph):
    """Locates a specific subgraph in the main graph"""
    # The atoms in the target structure - Only works with depth=1
    # Note: Place the anchor node as the first index!
    tag = np.array(['C', 'O', 'O', 'C'])
    # the adjacency matrix of the target structure
    adj = np.array([[0, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 0]])
    # The unique node types in the target structure
    unq = np.array(['C', 'O'])
    # Identify the edge to cut to separate the graphs
    cut = [0, 3]
    # The count of the different node types in the target
    cnt = {'C': 2, 'O': 2}
    cedges = [] # list of edges to cut
    for i, inode in enumerate(graph.nodes):
        # Is the node type the same as the target's anchor?
        if inode.atom != tag[0]:
            continue
        # Extract the list of connectivity for the node
        irow = graph.adj[i]
        # Check is the number of connections is correct
        if sum(irow) != len(tag) - 1:
            continue
        c_ids, good = [], True
        sc = {'%s' % inode.atom: 1}
        # Iterate over the adjacency vector for the node
        for k, kv in enumerate(graph.nodes):
            if irow[k] == 0:
                continue
            # Save the ID if the nodes are connected
            c_ids.append(k)
            # Check whether the node type is in the target structure
            if kv.atom not in unq:
                good = False
                break
            # Generate a count of the node types
            if kv.atom not in sc:
                sc[kv.atom] = 0
            sc[kv.atom] += 1
        # Check the node counts to ensure this substructure matches
        for atm in sc:
            if sc[atm] != cnt[atm]:
                good = False
        if not good:
            continue
        # If all the above checks are successful, the structure matches
        # Now we need to identify which bond to cut according to the "cut" array
        c_cnt = 0 # keeping a count, if there are multiple instances - code will flag
        c = []
        for j in c_ids:
            if graph.nodes[j].atom == tag[cut[1]]:
                c_cnt += 1
                c.append(j)
        if c_cnt > 1:
            print("Error: Multiple Cut Point Detected. Cannot Proceed")
            exit()
        ct = (i, j)
        cedges.append(ct)
    new_mat = []
    for i, irow in enumerate(graph.adj):
        nrow = []
        for j, ival in enumerate(irow):
            if (i, j) in cedges or (j, i) in cedges:
                nrow.append(0)
            else:
                nrow.append(ival)
        new_mat.append(nrow)
    x, y = cedges[0][0], cedges[0][1]
    new_mat = np.array(new_mat)
    #print_matrix(new_mat)


def trim_graph(adj, nodes, edges, save=False):
    """Trimming Procedure"""
    trimming, last = True, len(nodes)
    while trimming:
        adj, nodes, edges = reduce_graph(adj, nodes, edges, save=save)
        x = len(nodes)
        if x == last:
            trimming = False
        else:
            last = x
    return adj, nodes, edges


def find_loop(graph):
    """Attempts to find loops in a graph"""
    #print_matrix(graph.adj, tag='Full')
    # trim the graph to remove terminal atoms and terminal groups
    adj, nodes, edges = trim_graph(graph.adj, graph.nodes, graph.edges, save=True)

    #print_matrix(adj, tag='TrimmedFull')

    loops, anchors = [], []
    raw_sub = 0
    # Fine nodes where N >= 3 to use as anchors
    for i, node in enumerate(adj):
        if sum(node) < 3:
            continue
        anchors.append(i)
    # Set the largest loop to consider
    maxdepth, mindepth = MAX, MIN
    scnt = 0
    for i in anchors:
        depth = 1
        a_atm = nodes[i].label
        #if a_atm != 'C25':
        #    continue
        n, loop = sum(adj[i]), False
        v = list(reversed(np.argsort(adj[i])))[:n]
        nxt = v[:]
        while depth <= maxdepth: # and not loop:
            jnxt = []
            for j in nxt:
                jn = sum(adj[j])
                jv = list(reversed(np.argsort(adj[j])))[:jn]
                for jx in jv:
                    if loop:
                        pass
                    elif jx == i and depth >= mindepth:
                        loop = True
                    elif jx == i:
                        continue
                    else:
                        if jx not in v:
                            v.append(jx)
                        if jx not in jnxt:
                            jnxt.append(jx)
            depth += 1
            nxt = jnxt[:]
        if not loop:
            continue
        loopnodes = [i] + v
        loop_adj, loop_nodes, loop_edges = [], [], []
        for k, row in enumerate(adj):
            grow, edge = [], []
            if k not in loopnodes:
                continue
            loop_nodes.append(nodes[k])
            for l, val in enumerate(row):
                if l not in loopnodes:
                    continue
                grow.append(val)
                edge.append(edges[k, l])
            loop_adj.append(grow)
            loop_edges.append(edge)
        loop_adj = np.array(loop_adj)
        loop_nodes = np.array(loop_nodes)
        loop_edges = np.array(loop_edges)
        loop_adj, loop_nodes, loop_edges = trim_graph(loop_adj, loop_nodes, loop_edges)
        if len(loop_adj) > 0 and a_atm in [i.label for i in loop_nodes]:
            sub = (loop_adj, loop_nodes, loop_edges)
            #print_matrix(sub[0], tag='RawSub_%i' % raw_sub)
            sub = clip_graphs(sub, a_atm, scnt)
            raw_sub += 1
            loops.append(sub)
        scnt += 1
    return loops


def clip_graphs(sub, atm, scnt, N=3):
    """Clips subgraph, removing edges between nodes with >= N incident edges"""
    adj, nodes, edges = sub
    clips = []

    # Locate all nodes with >= N edges
    for i, row in enumerate(adj):
        if sum(row) < N:
            continue
        clips.append(i)
    if len(clips) < 2:
        return sub

    og_adj = adj[:] # backing up original adjacency matrix

    # Determine if any of those nodes are connected
    # Edited: Added a code to save a list of indices
    #         of clipped edges
    pairs, found = [], False
    cuts = []
    for i in clips:
        for j in clips:
            if i == j:
                continue
            # If they are connected, clip them
            if adj[i, j] == 1:
                adj[i, j], found = 0, True
                pair = (min([i, j]), max([i, j]))
                if pair not in cuts:
                    cuts.append(pair)

    seen = {} # list of nodes that have been seen before
    for cut in cuts:
        i, j = cut
        if i not in seen:
            seen[i] = 0
        if j not in seen:
            seen[j] = 0
        seen[i] += 1
        seen[j] += 1

    ncuts = []
    for cut in cuts:
        i, j = cut
        if seen[i] > 1 or seen[j] > 1:
            adj[i, j] = 1
            adj[j, i] = 1
        else:
            ncuts.append(cut)
    cuts = ncuts[:]

    if not found:
        return sub

    # Now determine which new cluster belongs to the anchor atom
    found = False
    for i, node in enumerate(nodes):
        if atm == node.label:
            found = True
            idx = i
    if not found:
        print('FATAL ERROR: Anchor atom not found in cluster')
        exit()
    n = sum(adj[idx])
    conns, last, running = [idx] + list(reversed(np.argsort(adj[idx])))[:n], None, True
    nxt = conns[:]
    cnt = 0
    while running:
        nnxt = []
        for j in nxt:
            jn = sum(adj[j])
            jnxt = list(reversed(np.argsort(adj[j])))[:jn]
            for x in jnxt:
                if x not in conns:
                    conns.append(x)
                    nnxt.append(x)
        if last is None:
            last = len(conns)
        elif len(conns) > last:
            last = len(conns)
        elif len(conns) == last:
            running = False
        cnt += 1
        nxt = nnxt[:]

    # If the size of the new node list is the same as the original subgraph:
    if len(conns) == len(nodes):
        return sub

    # If not build a new subgraph
    new_adj, new_nodes, new_edges = [], [], []
    for i, row in enumerate(adj):
        nrow = []
        erow = []
        if i not in conns:
            continue
        new_nodes.append(nodes[i])
        for j, val in enumerate(row):
            if j not in conns:
                continue
            pair = (min([i, j]), max([i, j]))
            nrow.append(adj[i, j])
            #if pair in cuts:
            #    nrow.append(1)
            #    print("Pairing:", pair)
            #else:
            #    nrow.append(adj[i, j])
            erow.append(edges[i, j])
        new_adj.append(nrow)
        new_edges.append(erow)
    new = (np.array(new_adj), np.array(new_nodes), np.array(new_edges))
    return new


def reduce_graph(g_adj, g_nodes, g_edges, N=1, save=False):
    """removes nodes with fewer than N nodes from the graph"""
    if save:
        global TERMS
    term = []
    for i, row in enumerate(g_adj):
        if sum(row) <= N:
            term.append(i)
    if save:
        #TERMS.append(term)
        tags = [g_nodes[i].label for i in term]
        TERMS.append(tags)
    adj, nodes, edges = [], [], []
    for i, row in enumerate(g_adj):
        arow = []
        erow = []
        if i in term:
            continue
        nodes.append(g_nodes[i])
        for j, val in enumerate(row):
            if j in term:
                continue
            try:
                arow.append(val)
                erow.append(g_edges[i, j])
            except TypeError:
                print(i, j)
                exit()
        adj.append(arow)
        edges.append(erow)
    return np.array(adj), np.array(nodes), np.array(edges)


def restore_terminals(graph, loops):
    """Restores the terminal atoms to the loops"""
    olayers = list(reversed(TERMS))
    layers, layer_ids = [], []
    for layer in olayers:
        if not len(layer):
            continue
        layers.append(layer)
        ids = []
        for i, node in enumerate(graph.nodes):
            if node.label in layer:
                ids.append(i)
        layer_ids.append(ids)
    new_loops = []
    for il, loop in enumerate(loops):
        adj, nodes, edges = loop
        ids = []
        for lnode in nodes:
            found = False
            for i, inode in enumerate(graph.nodes):
                if lnode.label == inode.label:
                    found = True
                    ids.append(i)
            if not found:
                print("FATAL ERROR: LOOP NOT NOT FOUND IN FULL GRAPH")
                exit()
        nlay = len(layers)
        # Identify which atoms in the clipped layer belonds
        # to the loop
        add = []
        full = ids[:]
        new = []
        for l, layer in enumerate(layers):
            if len(layer) == 0:
                continue
            ids += new[:]
            new = []
            for i, n in enumerate(layer_ids[l]):
                t = layer[i]
                for j in ids:
                    if graph.adj[n, j] == 1:
                        x = (i, n, t)
                        if x not in add:
                            add.append(x)
                            full.append(n)
                            new.append(n)
        new_adj, new_nodes, new_edges = [], [], []
        for i, row in enumerate(graph.adj):
            if i not in full:
                continue
            nrow, erow = [], []
            new_nodes.append(graph.nodes[i])
            for j, val in enumerate(row):
                if j not in full:
                    continue
                nrow.append(val)
                erow.append(graph.edges[i, j])
            new_adj.append(nrow)
            new_edges.append(erow)
        sub = (np.array(new_adj), np.array(new_nodes), np.array(new_edges))
        new_loops.append(sub)
    return new_loops


def redundant_chunks(chunks):
    """finds and removes redundant chunks"""
    cnts, used = [], []
    new_chunks = []
    kept = {}
    remove, chnk_ids = [], []
    for i, sub in enumerate(chunks):
        n, keep = len(sub[1]), True
        for a in sub[1]:
            if a.label in used:
                if n > kept[a][1]:
                    remove.append(kept[a][0])
                    kept[a] = (i, n)
                else:
                    keep = False
            else:
                used.append(a.label)
                kept[a] = (i, n)
        if keep:
            new_chunks.append(sub)
            chnk_ids.append(i)
    new2 = []
    for j, chunk in enumerate(new_chunks):
        i = chnk_ids[j]
        if i in remove:
            continue
        new2.append(chunk)
    return new2


def generate_tags(atoms):
    """Generates a tag identifier for each subplot"""
    atom_list = []
    atom_count = {}
    for atom in atoms:
        ele = atom_type(atom)
        if ele not in atom_list:
            atom_list.append(ele)
            atom_count[ele] = 0
        atom_count[ele] += 1
    atom_list.sort()
    tag = ''
    for atom in atom_list:
        tag += '%s%i' % (atom, atom_count[atom])
    return tag


def find_unique(subs):
    """counts the number of unique subgraphs in the MOF"""
    tags, cnts = [], []
    unique, frags = [], {}
    subplots = {}
    for sub in subs:
        adj, nodes, edges = sub
        atoms, n = [i.label for i in nodes], len(nodes)
        tag = generate_tags(atoms)
        if n not in cnts:
            unique.append(sub)
            cnts.append(n)
            tags.append(tag)
            frags[tag] = 1
        else:
            if tag not in tags:
                unique.append(sub)
                tags.append(tag)
                frags[tag] = 1
            else:
                frags[tag] += 1
        if tag not in subplots:
            subplots[tag] = []
        subplots[tag].append(sub)
    return unique, tags, frags, subplots


def dummy_nodes(subs, graph):
    """Generates a simplified graph"""
    unique, tags, frags, subplots = subs
    nodes = []
    # locate all connection points in the clusters
    for sub in subplots:
        for grph in subplots[sub]:
            atoms = [g.label for g in grph[1]]
            conns, gids, gnms = [], [], []
            for i, node in enumerate(graph.nodes):
                if node.label in atoms:
                    gids.append(i)
                    gnms.append(node.label)
            for i, gid in enumerate(gids):
                if sum(graph.adj[gid]) < 3:
                    continue
                n = sum(graph.adj[gid])
                v = list(reversed(np.argsort(graph.adj[gid])))[:n]
                m = [graph.nodes[k].label for k in v]
                inter = False
                for j in m:
                    if j not in gnms:
                        inter = True
                        toid = j
                if inter:
                    conns.append((gid, toid))
            nodes.append((sub, conns, grph, gnms))

    # ===============================================================
    # Note: From this point on, there is no more chemical info in
    #       the graphs
    # ===============================================================
    # Determine which nodes are connected to which nodes
    # by basically generating a new ajacency matrix
    new_adj = []
    for i, icls in enumerate(nodes):
        row = []
        isub, icn, igph, iatms = icls
        for j, jcls in enumerate(nodes):
            jsub, jcn, jgph, jatms = jcls
            found = False
            for cnn in icn:
                if cnn[1] in jatms:
                    found = True
            if found:
                row.append(1)
            else:
                row.append(0)
        new_adj.append(row)

    new_adj = np.array(new_adj)
    print(new_adj)
    #print_matrix(new_adj, tag='%s_DummyFull' % infile.split('.graph')[0])

    print('~' * 50)
    # To Do: Need to find a way to further reduce the plot. Right now
    # the super cell does not return the same graph as the unit cell
    clusters = []
    checked = []
    for i, row in enumerate(new_adj):
        if i in checked:
            continue
        checked.append(i)
        cluster, n = [i], sum(row)
        v = list(reversed(np.argsort(row)))[:n]
        cluster += v
        checked += v
        nxt, running, last = v[:], True, None
        while running:
            nnxt = []
            for j in nxt:
                jn = sum(new_adj[j])
                jv = list(reversed(np.argsort(new_adj[j])))[:jn]
                for jx in jv:
                    if jx not in cluster:
                        checked.append(jx)
                        cluster.append(jx)
                        nnxt.append(jx)
            if last is None:
                last = len(cluster)
            elif len(cluster) == last:
                running = False
            else:
                last = len(cluster)
            nxt = nnxt[:]
        clusters.append(cluster)
    if len(clusters) > 1:
        # Note I need to consider changing this later to maybe
        # select the LARGEST cluster
        cluster = clusters[0]
        a_adj, a_nodes = [], []
        for i, row in enumerate(new_adj):
            if i not in cluster:
                continue
            nrow = []
            a_nodes.append(nodes[i])
            for j, val in enumerate(row):
                if j not in cluster:
                    continue
                nrow.append(val)
            a_adj.append(nrow)
        new_adj = np.array(a_adj)
        nodes = np.array(a_nodes)
    #print_matrix(new_adj, tag='%s_Quotient' % infile.split('.graph')[0])
    dummy = (new_adj, nodes)
    return dummy


def find_ones(ilist):
    """Returns the indeces of the 1's in your list"""
    return list(reversed(np.argsort(ilist)))[:sum(ilist)]


def simplify(sub):
    """Further simplifies the subgraph"""
    new_adj, nodes = sub
    tags, unq = [], {}
    counts = {}

    # Generate full tag name for all nodes - incl connectivity
    nnodes = []
    for i, node in enumerate(nodes):
        tag = node[0]
        v = find_ones(new_adj[i])
        m_nodes = [nodes[i][0] for i in v]
        cnt, unique = {}, []
        for a in m_nodes:
            if a not in unique:
                unique.append(a)
                cnt[a] = 1
            else:
                cnt[a] += 1
        unique.sort()
        for i in unique:
            tag += '-%ix%s' % (cnt[i], i)
        nnodes.append(tag)
    nnodes = np.array(nnodes)

    # find all unique conformations of each cluster
    for i, node in enumerate(nodes):
        #tag = node[0]
        xii = nnodes[i]
        tag = '[' + nnodes[i] + ']'
        v = find_ones(new_adj[i])
        m_nodes = [nnodes[i] for i in v]
        cnt, unique = {}, []
        for a in m_nodes:
            if a not in unique:
                unique.append(a)
                cnt[a] = 1
            else:
                cnt[a] += 1
        unique.sort()
        for i in unique:
            tag += '-%ix[%s]' % (cnt[i], i)
        if tag not in tags:
            tags.append(tag)
            #unq[tag] = (node[0], unique, cnt, m_nodes)
            #print(unique)
            unq[tag] = (xii, unique, cnt, m_nodes)
            counts[tag] = 0
        counts[tag] += 1

    #print(counts)
    #exit()

    # determine the smallest possible structure based
    # on the unique configurations
    mini_adj, mini_nodes, row, ratios = [], [], [], {}

    # Find the smallest number - this will be used as the 1: in the ratios
    lowest, lowtag = None, None
    for tag in counts:
        if lowest is None:
            lowest = counts[tag]
            lowtag = tag
        elif lowest > counts[tag]:
            lowest=counts[tag]
            lowtag = tag

    fulllowtag = lowtag
    lowtag = lowtag.split(']')[0].split('[')[-1]
    print("Lowest:", lowtag, lowest, fulllowtag)

    # Now calculate the ratios relative to the lowest tag
    for tag in counts:
        jtag = tag.split(']')[0].split('[')[-1]
        if jtag == lowtag:
            continue
        ratio = counts[tag] / lowest
        ratios[tag] = ratio
#        print(lowtag + ':' + tag, '\t1:%i' % ratio)


    # Now determine the smallest number of lowtag that can exist
    # to generate a complete/representative subgraph
    min_cnt = None
    for tag in unq:
        inode, unique, cnt, m_nodes = unq[tag]
        if inode == lowtag:
            continue
        icount = 0
        #print(inode, m_nodes)
        for jnode in m_nodes:
            #print(jnode)
            if jnode == lowtag: #.split('-')[0]:
                icount += 1
        if min_cnt is None:
            min_cnt = icount
        elif icount > min_cnt:
            min_cnt = icount

    #print(min_cnt)
    #exit()

    # find the number of occurrences of smallest node
    # ignoring its connectivity
    #print('\n----------')
    icounts = {}
    for node in counts:
        inode = node.split(']')[0].split('[')[-1]
        if inode not in icounts:
            icounts[inode] = 0
        icounts[inode] += 1

    #print('~' * 25)
    sub_nodes = []
    for i in range(min_cnt):
        sub_nodes.append(fulllowtag)
    # adjust ratios accordingly
    new_ratios = {}
    for thing in ratios:
        ratio = ratios[thing]
        new = min_cnt * ratio
        if new != round(new):
            print('Error: Integer values not found! Code Improvements Needed to Adjust Ratios.')
            exit()
        new = int(new)
        new_ratios[thing] = new
        for i in range(new):
            sub_nodes.append(thing)

    #print('~' * 25)
    #print(sub_nodes, '\n')
    #print('New Nodes:')
    #for snode in sub_nodes:
    #    print('\t>>', snode.split('-')[0], '\t\t', snode.split('-')[1:])
    #print('\nTotal Nodes in Graph:', len(sub_nodes), '\n')
    #print(lowtag, min_cnt, new_ratios, '\n')

    #print('\n')

    # Now that we have the list of nodes found in our graph, we need
    # to determine the connectivity between those nodes

    # initialize the connectivity array - this will later be used
    # to build the adjacency matrix
    conns = [[] for i in range(len(sub_nodes))]
    #exit()
    numrs = [len(unq[tag][3]) for tag in sub_nodes]
    needs = [unq[tag][3][:] for tag in sub_nodes]

    #print(conns)
    for i, inode in enumerate(sub_nodes):
        #print('-' * 25)
        #x_inode = inode.split('-')[0]
        x_inode = inode.split(']')[0].split('[')[1]
        #print(i, x_inode)
        #print('Needs:', needs[i], '\n\n')
        for j, jnode in enumerate(sub_nodes):
            #x_jnode = jnode.split('-')[0]
            x_jnode = jnode.split(']')[0].split('[')[1]
            if i == j:
                continue
            x = jnode.split('-')[0]
            #if jnode.split('-')[0] in needs[i] and inode.split('-')[0] in needs[j]:
            if x_jnode in needs[i] and x_inode in needs[j]:
                conns[i].append(j)
                conns[j].append(i)
                #print(j, jnode, needs[j])
                needs[i].remove(x_jnode)
                needs[j].remove(x_inode)
                #print('\t', inode, needs[i])
                #print('\t', jnode, needs[j], '\n')
            #else:
            #    print('skipping:', x_jnode, '\t', needs[j])
    #print()
    #print(conns)
    #print('*' * 50)

    new_adj = []
    for i, inode in enumerate(sub_nodes):
        row = []
        for j, jnode in enumerate(sub_nodes):
            if i == j:
                row.append(0)
            else:
                if j in conns[i]:
                    row.append(1)
                else:
                    row.append(0)
        new_adj.append(row)
    new_adj, sub_nodes = np.array(new_adj), np.array(sub_nodes)
    new_sub = (new_adj, sub_nodes)
    return new_sub


def main():
    """Main execution of script"""
    os.system('cls')
    print('=' * 50 + '\nStarting Quotient Graph Code\n')
    print('Version %s' % VER)
    print('=' * 50)
    print("Loading graph data from", infile)
    print('~' * 50)
    if not os.path.exists(infile):
        print('Graph file: %s' % infile, "not found. Running graph generation")
        run = Popen(['python', 'graph.py', ciffile], stdout=PIPE)
        for line in run.stdout:
            #print(line.strip())
            pass
        print('~' * 50)
    else:
        print(infile, "file found... Loading graph...")
    cif, graph = pkl.load(open(infile, 'rb'))
    #print_matrix(graph.adj, tag='FullGraph')
    loops = find_loop(graph)
    for li, loop in enumerate(loops):
        tag = 'loop_%i' % li
        #print_matrix(loop[0], tag=tag)
    chnks = restore_terminals(graph, loops)
    chnks = redundant_chunks(chnks)
    subs = find_unique(chnks)
    print('Substructure Report:\n' + '-' * 50)
    for i, sub in enumerate(subs[0]):
        print('\n', subs[1][i], 'found', subs[2][subs[1][i]], 'time(s)')
        print(sub[0])
        print_matrix(sub[0], tag='%s_%s_Subgraph' % (infile.split('.graph')[0], subs[1][i]))
    print('=' * 50)
    print("Dummy Matrix\n")
    dummy = dummy_nodes(subs, graph)
    #print_matrix(dummy[0], tag='%s_DummyNodes' % infile.split('.graph')[0])
    simple = simplify(dummy)
    print("Simple Matrix\n")
    print(simple[0], '\n')
    print_matrix(simple[0], tag='%s_Quotient' % infile.split('.graph')[0])
    print('Node Legend:\n')
    for i, node in enumerate(simple[1]):
        print('>', i, '=', node)
    print('\n' + '=' * 50)
    print('Program Terminated Normally.\n')


if __name__ in '__main__':
    main()
