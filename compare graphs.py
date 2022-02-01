#!/usr/bin/env python

"""
Compares two graphs for a unit cell and supercell of the same MOF
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import itertools as it
from graph import Cif, Node, Edge, Graph


unit = 'ZnBDC.graph'
supr = 'ZnBDC_Super.graph'


def print_distance(graph):
    """Prints the distance between two non zero points in the adjacency matrix"""
    counts, c = [], 0
    print('=' * 50)
    print(graph.incidence)
    print(graph.incidence.shape)
    for i, val in enumerate(graph.incidence[0]):
        if val != 0:
            counts.append(c)
            c = 0
        else:
            c += 1
    print(counts)


def main():
    """Main execution of script"""
    ucif, ugraph = pkl.load(open(unit, 'rb'))
    scif, sgraph = pkl.load(open(supr, 'rb'))
    print_distance(ugraph)
    print_distance(sgraph)
    ushp, sshp = ugraph.incidence.shape, sgraph.incidence.shape
    print('*' * 50)
    print(ushp)
    print(sshp)
    print(sshp[0] / ushp[0], sshp[1] / ushp[1])


if __name__ in '__main__':
    main()
