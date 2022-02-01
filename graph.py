"""
Generates Descriptors according to protocol set in:

DOI: 10.1038/ncomms15679
"""


import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import itertools as it


RERUN = True
PROP  = 'EN'


class Cif(object):

    def __init__(self, ifile):
        self.name  = None
        self.atoms = None
        self.conn  = None
        self.cell  = None
        self.r     = None
        self.natoms = None
        self.volume = None
        self.import_cif(ifile)

    def set_atoms(self, atoms):
        self.atoms = atoms
        self.natoms = len(atoms)

    def import_cif(self, ifile):
        """Imports all of the data from the cif"""
        self.name = ifile.split('\\')[-1].split('.cif')[0]
        raw, atoms, conns = open(ifile, 'r').readlines(), {}, {}
        start, trig1, trig2 = False, '_atom_type_partial_charge', '_atom_site_occupancy'
        cstart, ctrig = False, '_ccdc_geom_bond_type'
        cell = {'_cell_length_a'   : None,
                '_cell_length_b'   : None,
                '_cell_length_c'   : None,
                '_cell_angle_alpha': None,
                '_cell_angle_beta' : None,
                '_cell_angle_gamma': None,
                '_cell_volume'     : None}
        for line in raw:
            if not start and trig1 in line:
                start = True
                tids = (-4, -1)
            elif not start and trig2 in line:
                start = True
                tids = (-6, -3)
            elif not cstart and ctrig in line:
                cstart = True
            elif start and (len(line.strip()) == 0 or 'loop_' in line):
                start = False
            elif start:
                line = [val.strip() for val in line.split()]
                tag, ele = line[0], line[1]
                coord = np.array([float(val) for val in line[tids[0]: tids[1]]])
                atoms[tag] = (ele, coord)
            elif len(line.split()) > 1 and not start and not cstart:
                if line.split()[0].strip() in cell:
                    cell[line.split()[0].strip()] = float(line.split()[-1].strip())
            elif cstart and (len(line.strip()) == 0 or 'loop_' in line):
                cstart = False
            elif cstart:
                line = [val.strip() for val in line.split()]
                one, two, lgt, typ = line[0], line[1], float(line[2]), line[4]
                if one not in conns:
                    conns[one] = []
                if two not in conns:
                    conns[two] = []
                if (two, lgt, typ) not in conns[one]:
                    conns[one].append((two, lgt, typ))
                if (one, lgt, typ) not in conns[two]:
                    conns[two].append((one, lgt, typ))
        self.set_atoms(atoms)
        self.conn = conns
        self.cell = cell
        a = cell['_cell_length_a']
        b = cell['_cell_length_b']
        c = cell['_cell_length_c']
        alpha = np.deg2rad(cell['_cell_angle_alpha'])
        beta = np.deg2rad(cell['_cell_angle_beta'])
        gamma = np.deg2rad(cell['_cell_angle_gamma'])
        cosa = np.cos(alpha)
        cosb = np.cos(beta)
        cosg = np.cos(gamma)
        vol1 = (1 - cosa**2 - cosb**2 - cosg**2)
        vol2 = 2 * (cosa*cosb*cosg)
        self.volume = a*b*c * (vol1 + vol2) ** 0.5
        self.r = setup_carconv(cell)


class Graph(object):

    def __init__(self, cif):
        self.nodes     = None
        self.edges     = None
        self.adj       = None
        self.galvez    = None
        self.distance  = None
        self.reci      = None
        self.incidence = None
        self.build_graph(cif)

    def build_graph(self, cif):
        """Builds the total graph"""
        nodes, edges = [], []
        for tag in tqdm(cif.atoms):
            row = []
            cto = [val[0] for val in cif.conn[tag]]
            lns = [val[1] for val in cif.conn[tag]]
            tps = [val[2] for val in cif.conn[tag]]
            node = Node()
            node.atom = cif.atoms[tag][0]
            node.set_coords(cif.atoms[tag][1], cif)
            node.label = tag
            for jtag in tqdm(cif.atoms):
                if tag == jtag:
                    row.append(None)
                    continue
                elif jtag in cto:
                    edge = Edge()
                    j = cto.index(jtag)
                    edge.length = lns[j]
                    edge.type = tps[j]
                    row.append(edge)
                else:
                    row.append(None)
            edges.append(row)
            nodes.append(node)
        self.nodes = np.array(nodes)
        self.edges = np.array(edges)
        adj = []
        print('\n\t>> Done.')
        print('\n  Generating Adjacency Graph')
        for row in tqdm(edges):
            nrw, grw = [], []
            for val in tqdm(row):
                if val is None:
                    nrw.append(0)
                else:
                    nrw.append(1)
            adj.append(nrw)
        self.adj = np.array(adj)
        print('\n\t>> Done.')
        #print('\n  Generating Distance Matrices')
        #self.distance_matrix()
        #print('\n\t>> Done.')
        #print('\n  Generating Galvez Matrix')
        #self.galvez = np.matmul(self.adj, self.reci)
        #print('\t>> Done.')
        #print('\n  Generating Incidence Graph')
        #self.incidence_graph()
        #print('\t>> Done.')

    def distance_matrix(self):
        """Calculates all of the distances"""
        matr, reci = [], []
        for i, inode in tqdm(enumerate(self.nodes)):
            row, rrow = [], []
            for j, jnode in tqdm(enumerate(self.nodes)):
                combos = list(it.product(inode.ccoord, jnode.ccoord))
                combos = np.array([np.array([com[0], com[1]]) for com in combos])
                combo = combos[0]
                d = min([dist(x[0], x[1]) for x in combos])
                if d == 0:
                    rrow.append(0)
                else:
                    rrow.append(1 / (d ** 2))
                row.append(d)
            matr.append(row)
            reci.append(rrow)
        self.distance = np.array(matr)
        self.reci = np.array(reci)

    def incidence_graph(self):
        """Calculates the undirected incidence graph"""
        _edges, _used = [], []
        for i, iedge in enumerate(self.edges):
            for j, jedge in enumerate(iedge):
                if jedge is None:
                    continue
                _etag = '%i-%i' % (min(i, j), max(i, j))
                if _etag in _used:
                    continue
                _edges.append((i, j, jedge))
                _used.append(_etag)
        #self.edges = _edges
        incid = []
        for i, node in enumerate(self.nodes):
            row = []
            for j, _edge in enumerate(_edges):
                if i in _edge:
                    row.append(1)
                else:
                    row.append(0)
            incid.append(row)
        self.incidence = np.array(incid)


class Node(object):

    def __init__(self):
        self.atom   = None
        self.label  = None
        self.ccoord = None
        self.fcoord = None

    def set_coords(self, fcoord, cell):
        """sets the carterisan coordinate for the node"""
        self.fcoord = fcoord
        self.ccoord = self.get_carts(fcoord, cell)

    def get_carts(self, _fsite, cif):
        """Generates a list of cartesian coordinates for the node"""
        _fsite = np.array(_fsite)
        _site_list = [_fsite]
        _arr = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                         [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
                         [1., 1., 1.]])
        for _a in _arr:
            _site_list.append(_fsite - _a)
            _site_list.append(_fsite + _a)
        _asite = []
        for _site in _site_list:
            _asite.append(frac_to_cart(_site, cif.r))
        return _asite


class Edge(object):

    def __init__(self):
        self.type = None
        self.length = None

    def set_length(self, coord1, coord2):
        """Determines the length of the edge given"""
        pass


def frac_to_cart(coord, r):
    """Converts a fractional coord to cartesian"""
    frac = np.zeros((3, 1))
    frac[0, 0] = coord[0]
    frac[1, 0] = coord[1]
    frac[2, 0] = coord[2]
    cart = np.matmul(r, frac)
    x_cart = cart[0, 0]
    y_cart = cart[1, 0]
    z_cart = cart[2, 0]
    return [x_cart, y_cart, z_cart]


def dist(one, two):
    """Calculates the euclidean distance"""
    return sum(((one - two) ** 2)) ** 0.5


def setup_carconv(cell):
    """Sets up the matrix to convert fractional to cartesian coords"""
    a_cell = cell['_cell_length_a']
    b_cell = cell['_cell_length_b']
    c_cell = cell['_cell_length_c']

    alpha  = cell['_cell_angle_alpha']
    alpha  = np.deg2rad(alpha)
    beta   = cell['_cell_angle_beta']
    beta   = np.deg2rad(beta)
    gamma  = cell['_cell_angle_gamma']
    gamma  = np.deg2rad(gamma)

    cosa   = np.cos(alpha)
    sina   = np.sin(alpha)
    cosb   = np.cos(beta)
    sinb   = np.sin(beta)
    cosg   = np.cos(gamma)
    sing   = np.sin(gamma)

    volume = 1.0 - cosa ** 2.0 - cosb ** 2.0 - cosg ** 2.0 + 2.0 * cosa * cosb * cosg
    volume = volume ** 0.5

    # Generate the Conversion Matrix
    r = np.zeros((3, 3))
    r[0, 0] = a_cell
    r[0, 1] = b_cell * cosg
    r[0, 2] = c_cell * cosb
    r[1, 1] = b_cell * sing
    r[1, 2] = c_cell * (cosa - cosb * cosg) / sing
    r[2, 2] = c_cell * volume / sing
    return r


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


def import_props(ap):
    """Imports the properties used in the descriptor calculation"""
    prop, dat = {}, pd.read_csv('Properties\\' + ap + '.csv')
    for i in dat.index:
        prop[dat['atom'][i]] = dat['value'][i]
    return prop


def descriptor(prop, graph):
    """Calculates the descriptor TE from the paper"""
    te = 0
    for i, inode in enumerate(graph.nodes):
        for j, jnode in enumerate(graph.nodes):
            d = abs(prop[inode.atom] - prop[jnode.atom])
            m = graph.galvez[i, j]
            te += (d * m)
    return te


def main():
    """Main execution"""
    if not os.path.exists(ofile) or RERUN:
        print("\n  Importing Cif Data")
        print('\t>> Done.')
        cif = Cif(ciffile)
        print("\n  Generating Graph")
        graph = Graph(cif)
        print('\t>>: Graph Calculations Complete.')
        print("\n  Dumping graph to", ofile)
        pkl.dump((cif, graph), open(ofile, 'wb'))
    else:
        print("\n  Importing Previously Calculated Graph.\n")
        cif, graph = pkl.load(open(ofile, 'rb'))
    if cif.natoms == 0:
        print('Error: No Atoms Identified - cif file may be in unrecognized format.')
        exit()
    print(graph.edges.shape)
    exit()
    props = import_props(PROP)
    te = descriptor(props, graph)
    print(te, cif.natoms, cif.volume)


if __name__ in '__main__':
    script, ciffile = sys.argv
    ofile = ciffile.split('.cif')[0].split('\\')[-1] + '.graph'
    main()
