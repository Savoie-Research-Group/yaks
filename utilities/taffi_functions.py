import sys,argparse,os,time,math,subprocess,timeit
import random
import ast
import collections
import numpy as np
from scipy.spatial.distance import cdist
from copy import copy,deepcopy
import itertools
from rdkit.Chem import AllChem,rdchem,BondType,MolFromSmiles,Draw,Atom,AddHs,HybridizationType
# Generates the adjacency matrix based on UFF bond radii
# Inputs:       Elements: N-element List strings for each atom type
#               Geometry: Nx3 np.array holding the geometry of the molecule
#               File:  Optional. If Table_generator encounters a problem then it is often useful to have the name of the file the geometry came from printed. 
def count_bond_matrix(elements, bond_matrix):
    if not hasattr(find_lewis, "lone_e"):

        # Used for determining number of valence electrons provided by each atom to a neutral molecule when calculating Lewis structures
        find_lewis.valence = {  'h':1, 'he':2,\
                                   'li':1, 'be':2,                                                                                                                'b':3,  'c':4,  'n':5,  'o':6,  'f':7, 'ne':8,\
                                   'na':1, 'mg':2,                                                                                                               'al':3, 'si':4,  'p':5,  's':6, 'cl':7, 'ar':8,\
                                    'k':1, 'ca':2, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':3, 'ge':4, 'as':5, 'se':6, 'br':7, 'kr':8,\
                                   'rb':1, 'sr':2,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':3, 'sn':4, 'sb':5, 'te':6,  'i':7, 'xe':8,\
                                   'cs':1, 'ba':2, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':3, 'pb':4, 'bi':5, 'po':6, 'at':7, 'rn':8  }        
        find_lewis.atomic_to_element = { find_lewis.periodic[i]:i for i in find_lewis.periodic.keys() }
    elements_lower = [ _.lower() for _ in elements ]
    valence = np.array([ find_lewis.valence[_] for _ in elements_lower ])
    lones=[]
    charges=[]
    bonding_electrons=[]
    core_electrons=[]
    # check the dimension
    if len(np.array(bond_matrix).shape)==3:
        for b_mat in bond_matrix:
            lone=[]
            charge=[]
            bond=[]
            core=[]
            for count_i, i in enumerate(b_mat):
                l=0
                b=0
                c=0
                for count_j, j in enumerate(i):
                    if count_i==count_j:
                        l=l+j
                        c=c+j
                    elif j!=0:
                        b=b+j
                        c=c+j
                lone.append(int(l))
                bond.append(int(b))
                core.append(int(c))
            lones.append(lone)
            charges.append(return_formals(b_mat, [_.lower() for _ in elements]))
            bonding_electrons.append(bond)
            core_electrons.append(core)
    else:
        for count_i, i in enumerate(bond_matrix):
            l=0
            b=0
            c=0
            for count_j, j in enumerate(i):
                if count_i==count_j:
                    l=l+j
                    c=c+j
                elif j!=0:
                    b=b+j
                    c=c+j
            lones.append(int(l))
            bonding_electrons.append(int(b))
            core_electrons.append(int(c))
        charges.append(return_formals(bond_matrix, [_.lower() for _ in elements]))
    return lones, charges, bonding_electrons, core_electrons
def Table_generator(Elements,Geometry,File=None):

    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # NOTE: Units of angstroms 
    # NOTE: These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist for the same atom, the largest was used. 
    Radii = {  'H':0.354, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.244, 'Si':1.117,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # SAME AS ABOVE BUT WITH A SMALLER VALUE FOR THE Al RADIUS ( I think that it tends to predict a bond where none are expected
    Radii = {  'H':0.39, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.15,  'Si':1.050,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.400, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # Use Radii json file in Lib folder if sepcified
    Max_Bonds = {  'H':2,    'He':1,\
                  'Li':None, 'Be':None,                                                                                                                'B':4,     'C':4,     'N':4,     'O':2,     'F':1,    'Ne':1,\
                  'Na':None, 'Mg':None,                                                                                                               'Al':4,    'Si':4,  'P':None,  'S':None, 'Cl':None,    'Ar':1,\
                   'K':None, 'Ca':None, 'Sc':15, 'Ti':14,  'V':13, 'Cr':12, 'Mn':11, 'Fe':10, 'Co':9, 'Ni':8, 'Cu':None, 'Zn':None, 'Ga':3,    'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                  'Rb':None, 'Sr':None,  'Y':15, 'Zr':14, 'Nb':13, 'Mo':12, 'Tc':11, 'Ru':10, 'Rh':9, 'Pd':8, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                  'Cs':None, 'Ba':None, 'La':15, 'Hf':14, 'Ta':13,  'W':12, 'Re':11, 'Os':10, 'Ir':9, 'Pt':8, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }
                     
    # Scale factor is used for determining the bonding threshold. 1.2 is a heuristic that give some lattitude in defining bonds since the UFF radii correspond to equilibrium lengths. 
    scale_factor = 1.2

    # Print warning for uncoded elements.
    for i in Elements:
        if i not in Radii.keys():
            print( "ERROR in Table_generator: The geometry contains an element ({}) that the Table_generator function doesn't have bonding information for. This needs to be directly added to the Radii".format(i)+\
                  " dictionary before proceeding. Exiting...")
            quit()

    # Generate distance matrix holding atom-atom separations (only save upper right)
    Dist_Mat = np.triu(cdist(Geometry,Geometry))
    
    # Find plausible connections
    x_ind,y_ind = np.where( (Dist_Mat > 0.0) & (Dist_Mat < max([ Radii[i]**2.0 for i in Radii.keys() ])) )

    # Initialize Adjacency Matrix
    Adj_mat = np.zeros([len(Geometry),len(Geometry)])

    # Iterate over plausible connections and determine actual connections
    for count,i in enumerate(x_ind):
        
        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*scale_factor:            
            Adj_mat[i,y_ind[count]]=1
    
        if Elements[i] == 'H' and Elements[y_ind[count]] == 'H':
            if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*1.5:
                Adj_mat[i,y_ind[count]]=1

    # Hermitize Adj_mat
    Adj_mat=Adj_mat + Adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = { i:0 for i in Radii.keys() }
    conditions = { "H":1, "C":4, "F":1, "Cl":1, "Br":1, "I":1, "O":2, "N":4, "B":4 }
    for count_i,i in enumerate(Adj_mat):

        if Max_Bonds[Elements[count_i]] is not None and sum(i) > Max_Bonds[Elements[count_i]]:
            problem_dict[Elements[count_i]] += 1
            cons = sorted([ (Dist_Mat[count_i,count_j],count_j) if count_j > count_i else (Dist_Mat[count_j,count_i],count_j) for count_j,j in enumerate(i) if j == 1 ])[::-1]
            while sum(Adj_mat[count_i]) > Max_Bonds[Elements[count_i]]:
                sep,idx = cons.pop(0)
                Adj_mat[count_i,idx] = 0
                Adj_mat[idx,count_i] = 0

    # Print warning messages for obviously suspicious bonding motifs.
    if sum( [ problem_dict[i] for i in problem_dict.keys() ] ) > 0:
        print( "Table Generation Warnings:")
        for i in sorted(problem_dict.keys()):
            if problem_dict[i] > 0:
                if File is None:
                    if i == "H": print( "WARNING in Table_generator: {} hydrogen(s) have more than one bond.".format(problem_dict[i]))
                    if i == "C": print( "WARNING in Table_generator: {} carbon(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "Si": print( "WARNING in Table_generator: {} silicons(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "F": print( "WARNING in Table_generator: {} fluorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Cl": print( "WARNING in Table_generator: {} chlorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Br": print( "WARNING in Table_generator: {} bromine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "I": print( "WARNING in Table_generator: {} iodine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "O": print( "WARNING in Table_generator: {} oxygen(s) have more than two bonds.".format(problem_dict[i]))
                    if i == "N": print( "WARNING in Table_generator: {} nitrogen(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "B": print( "WARNING in Table_generator: {} bromine(s) have more than four bonds.".format(problem_dict[i]))
                else:
                    if i == "H": print( "WARNING in Table_generator: parsing {}, {} hydrogen(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "C": print( "WARNING in Table_generator: parsing {}, {} carbon(s) have more than four bonds.".format(File,problem_dict[i]))
                    if i == "Si": print( "WARNING in Table_generator: parsing {}, {} silicons(s) have more than four bonds.".format(File,problem_dict[i]))
                    if i == "F": print( "WARNING in Table_generator: parsing {}, {} fluorine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "Cl": print( "WARNING in Table_generator: parsing {}, {} chlorine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "Br": print( "WARNING in Table_generator: parsing {}, {} bromine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "I": print( "WARNING in Table_generator: parsing {}, {} iodine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "O": print( "WARNING in Table_generator: parsing {}, {} oxygen(s) have more than two bonds.".format(File,problem_dict[i]))
                    if i == "N": print( "WARNING in Table_generator: parsing {}, {} nitrogen(s) have more than four bonds.".format(File,problem_dict[i]))
                    if i == "B": print( "WARNING in Table_generator: parsing {}, {} bromine(s) have more than four bonds.".format(File,problem_dict[i]))
        print( "")

    return Adj_mat
# Generate model compounds using bond-electron matrix with truncated graph and homolytic bond cleavages
def generate_model_compound(index):
    print("this isn't coded yet")

# New find lewis algorithm
# To do: some speedups might be possible with sparse arrays instead of dense bond_mats for larger molecules.
# elements (list) : list of element labels
# adj_mat (array) : 2d array of atom connectivity (no bond orders)    
# q (int) : charge
# rings (list) : list of lists holding the atom indices in each ring. If none, then the rings are calculated
# mats_max (int) : the maximum number of bond electron matrices to return
# mats_thresh (float) : the value used to determine if a bond electron matrix is worth returning to the user. Any matrix with a score within this value of the minimum structure will be returned as a potentially relevant resonance structure (up to mats_max). 
# w_def (float) : weight of the electron deficiency term in the objective function for scoring bmats
# w_formal (float) : weight of the formal charge term in the objective function for scoring bmats
# w_aro (float) : weight of the aromatic term in the objective function for scoring bmats
# w_rad (float) : weight of the radical term in the objective function for scoring bmats
# local_opt (boolean) : controls whether non-local charge transfers are allowed (False). This can be expensive. (default: True)
def find_lewis(elements,adj_mat,q=0,rings=None,mats_max=10,mats_thresh=0.5,w_def=-0.7,w_exp=0.1,w_formal=0.1,w_aro=-24,w_rad=0.1,local_opt=True):
    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(find_lewis, "lone_e"):

        # Used for determining number of valence electrons provided by each atom to a neutral molecule when calculating Lewis structures
        find_lewis.valence = {  'h':1, 'he':2,\
                                   'li':1, 'be':2,                                                                                                                'b':3,  'c':4,  'n':5,  'o':6,  'f':7, 'ne':8,\
                                   'na':1, 'mg':2,                                                                                                               'al':3, 'si':4,  'p':5,  's':6, 'cl':7, 'ar':8,\
                                    'k':1, 'ca':2, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':3, 'ge':4, 'as':5, 'se':6, 'br':7, 'kr':8,\
                                   'rb':1, 'sr':2,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':3, 'sn':4, 'sb':5, 'te':6,  'i':7, 'xe':8,\
                                   'cs':1, 'ba':2, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':3, 'pb':4, 'bi':5, 'po':6, 'at':7, 'rn':8  }        

        # Used for determining electron deficiency when calculating lewis structures
        find_lewis.n_electrons = {  'h':2, 'he':2,\
                                       'li':0, 'be':0,                                                                                                                'b':8,  'c':8,  'n':8,  'o':8,  'f':8, 'ne':8,\
                                       'na':0, 'mg':0,                                                                                                               'al':8, 'si':8,  'p':8,  's':8, 'cl':8, 'ar':8,\
                                        'k':0, 'ca':0, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':8, 'ge':8, 'as':8, 'se':8, 'br':8, 'kr':8,\
                                       'rb':0, 'sr':0,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':8, 'sn':8, 'sb':8, 'te':8,  'i':8, 'xe':8,\
                                       'cs':0, 'ba':0, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':8, 'pb':8, 'bi':8, 'po':8, 'at':8, 'rn':8  }        

        # Used to determine is expanded octets are allowed when calculating Lewis structures
        find_lewis.expand_octet = { 'h':False, 'he':False,\
                                       'li':False, 'be':False,                                                                                                               'b':False,  'c':False, 'n':False, 'o':False, 'f':False,'ne':False,\
                                       'na':False, 'mg':False,                                                                                                               'al':True, 'si':True,  'p':True,  's':True, 'cl':True, 'ar':True,\
                                        'k':False, 'ca':False, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':True, 'ge':True, 'as':True, 'se':True, 'br':True, 'kr':True,\
                                       'rb':False, 'sr':False,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':True, 'sn':True, 'sb':True, 'te':True,  'i':True, 'xe':True,\
                                       'cs':False, 'ba':False, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':True, 'pb':True, 'bi':True, 'po':True, 'at':True, 'rn':True  }
        
        # Electronegativity (Allen scale)
        find_lewis.en = { "h" :2.3,  "he":4.16,\
                          "li":0.91, "be":1.58,                                                                                                               "b" :2.05, "c" :2.54, "n" :3.07, "o" :3.61, "f" :4.19, "ne":4.79,\
                          "na":0.87, "mg":1.29,                                                                                                               "al":1.61, "si":1.91, "p" :2.25, "s" :2.59, "cl":2.87, "ar":3.24,\
                          "k" :0.73, "ca":1.03, "sc":1.19, "ti":1.38, "v": 1.53, "cr":1.65, "mn":1.75, "fe":1.80, "co":1.84, "ni":1.88, "cu":1.85, "zn":1.59, "ga":1.76, "ge":1.99, "as":2.21, "se":2.42, "br":2.69, "kr":2.97,\
                          "rb":0.71, "sr":0.96, "y" :1.12, "zr":1.32, "nb":1.41, "mo":1.47, "tc":1.51, "ru":1.54, "rh":1.56, "pd":1.58, "ag":1.87, "cd":1.52, "in":1.66, "sn":1.82, "sb":1.98, "te":2.16, "i" :2.36, "xe":2.58,\
                          "cs":0.66, "ba":0.88, "la":1.09, "hf":1.16, "ta":1.34, "w" :1.47, "re":1.60, "os":1.65, "ir":1.68, "pt":1.72, "au":1.92, "hg":1.76, "tl":1.79, "pb":1.85, "bi":2.01, "po":2.19, "at":2.39, "rn":2.60} 

        # Polarizability ordering (for determining lewis structure)
        find_lewis.pol ={ "h" :4.5,  "he":1.38,\
                          "li":164.0, "be":377,                                                                                                               "b" :20.5, "c" :11.3, "n" :7.4, "o" :5.3,  "f" :3.74, "ne":2.66,\
                          "na":163.0, "mg":71.2,                                                                                                              "al":57.8, "si":37.3, "p" :25.0,"s" :19.4, "cl":14.6, "ar":11.1,\
                          "k" :290.0, "ca":161.0, "sc":97.0, "ti":100.0, "v": 87.0, "cr":83.0, "mn":68.0, "fe":62.0, "co":55, "ni":49, "cu":47.0, "zn":38.7,  "ga":50.0, "ge":40.0, "as":30.0,"se":29.0, "br":21.0, "kr":16.8,\
                          "rb":320.0, "sr":197.0, "y" :162,  "zr":112.0, "nb":98.0, "mo":87.0, "tc":79.0, "ru":72.0, "rh":66, "pd":26.1, "ag":55, "cd":46.0,  "in":65.0, "sn":53.0, "sb":43.0,"te":28.0, "i" :32.9, "xe":27.3,}

        # Initialize periodic table
        find_lewis.periodic = { "h": 1,  "he": 2,\
                                 "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                                 "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                  "k":19, "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                                 "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                                 "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

        
        # Initialize periodic table
        find_lewis.atomic_to_element = { find_lewis.periodic[i]:i for i in find_lewis.periodic.keys() }

    # Can be removed in lower-case centric future
    elements_lower = [ _.lower() for _ in elements ]
    #print('elements in find_lewis: ', elements_lower)
    # Array of atom-wise electroneutral electron expectations for convenience.
    eneutral = np.array([ find_lewis.valence[_] for _ in elements_lower ])    
    #print('eneutral in find_lewis: ', eneutral)
    # Array of atom-wise octet requirements for determining electron deficiencies
    e_tet = np.array([ find_lewis.n_electrons[_] for _ in elements_lower ])
    #print('e_tet in find lewis', e_tet)
    q = int(q)
    # Check that there are enough electrons to at least form all sigma bonds consistent with the adjacency
    #if ( sum(eneutral) - q  < sum( adj_mat[np.triu_indices_from(adj_mat,k=1)] )*2.0 ):
    #    print("ERROR: not enough electrons to satisfy minimal adjacency requirements")

    # Generate rings if they weren't supplied. Needed to determine allowed double bonds in rings and resonance
    if not rings: rings = return_rings(adjmat_to_adjlist(adj_mat),max_size=10,remove_fused=True)

    # Get the indices of atoms in rings < 10 (used to determine if multiple double bonds and alkynes are allowed on an atom)
    ring_atoms = { j for i in [ _ for _ in rings if len(_) < 10 ] for j in i }

    # Get the indices of bridgehead atoms whose largest parent ring is smaller than 8 (i.e., Bredt's rule says no double-bond can form at such bridgeheads)
    bredt_rings = [ set(_) for _ in rings if len(_) < 8 ]
    bridgeheads = []
    if len(bredt_rings) > 2:
        for r in itertools.combinations(bredt_rings,3):
            bridgeheads += list(r[0].intersection(r[1].intersection(r[2]))) # bridgeheads are atoms in at least three rings. 
    bridgeheads = set(bridgeheads)

    # Get the graph separations if local_opt = True
    if local_opt:
        seps = graph_seps(adj_mat)
    # using seps=0 is equivalent to allowing all charge transfers (i.e., all atoms are treated as nearby)
    else:
        seps = np.zeros([len(elements),len(elements)])
    #print('seps: ', seps)
    # Initialize lists to hold bond_mats and scores
    bond_mats = []
    scores = []

    # Initialize score function for ranking bond_mats
    en = np.array([ find_lewis.en[_] for _ in elements_lower ]) # base electronegativities of each atom
    #print('electronegativities: ', en)
    # Dangerous change by Hsuan-Hao
    if q:
        w_def=-0.15*w_formal
    # it's for generate the correct smiles and charge state for lithium ion study
    factor = -min(en)*q*w_formal if q>=0 else -max(en)*q*w_formal # subtracts off trivial formal charge penalty from cations and anions so that they have a baseline score of 0 all else being equal. 
    obj_fun = lambda x: bmat_score(x,elements_lower,rings,cat_en=en,an_en=en,rad_env=0*en,e_tet=e_tet,\
              w_def=w_def,w_exp=w_exp,w_formal=w_formal,w_aro=0,w_rad=w_rad,factor=factor,verbose=False) # aro term is turned off initially since it traps greedy optimization
    #print('here just at the beginning of the for loop')
    # Find the minimum bmat structure
    # gen_init() generates a series of initial guesses. For neutral molecules, this guess is singular. For charged molecules, it will yield all possible charge placements (expensive but safe). 
    for score,bond_mat,reactive in gen_init(obj_fun,adj_mat,elements_lower,rings,q):
        #print('here in the for loop')
        if bmat_unique(bond_mat,bond_mats):
            scores += [score]
            bond_mats += [bond_mat]
            sys.setrecursionlimit(5000)            
            bond_mats,scores,_,_ = gen_all_lstructs(obj_fun,bond_mats, scores, elements_lower, reactive, rings, ring_atoms, bridgeheads, seps=np.zeros([len(elements),len(elements)]), min_score=scores[0], ind=len(bond_mats)-1,N_score=1000,N_max=10000,min_opt=True)
    # Update objective function to include (anti)aromaticity considerations and update scores of the current bmats
    obj_fun = lambda x: bmat_score(x,elements_lower,rings,cat_en=en,an_en=en,rad_env=0*en,e_tet=e_tet,w_def=w_def,w_exp=w_exp,w_formal=w_formal,w_aro=w_aro,w_rad=w_rad,factor=factor,verbose=False)                        
    scores = [ obj_fun(_) for _ in bond_mats ]               
    # Sort by initial scores
    bond_mats = [ _[1] for _ in sorted(zip(scores,bond_mats),key=lambda x:x[0]) ] # sorting based on the scores and taking the second element (bond mats)
    scores = sorted(scores)
    #print('scores: ', scores[0])
    #print('bond matrices: ', bond_mats[0])
    #print('----------------')

    # Generate resonance structures: Run starting from the minimum structure and allow moves that are within s_window of the min_enegy score
    bond_mats = [bond_mats[0]]
    scores = [scores[0]]
    bond_mats,scores,_,_ = gen_all_lstructs(obj_fun,bond_mats, scores, elements_lower, reactive, rings, ring_atoms, bridgeheads, seps, min_score=scores[0], ind=len(bond_mats)-1,N_score=1000,N_max=10000,min_opt=False,min_win=0.5)
    #print(bond_mats[0])
    # Sort by initial scores
    bond_mats = [ _[1] for _ in sorted(zip(scores,bond_mats),key=lambda x:x[0]) ]
    scores = sorted(scores)
    N_gen = len(bond_mats)
    # Keep all bond-electron matrices within mats_thresh of the minimum but not more than mats_max total
    flag = True
    for count,i in enumerate(scores):
        if count > mats_max-1:
            flag = False
            break
        if i - scores[0] < mats_thresh:
            continue
        else:
            flag = False
            break
    if flag:
        count += 1

    # Shed the excess b_mats
    bond_mats = bond_mats[:count]
    scores = scores[:count]
    # Calculate the number of charge centers bonded to each atom (determines hybridization)
    # calculated as: number of bonded_atoms + number of unbound electron orbitals (pairs or radicals).
    # The latter is calculated as the minimum value over all relevant bond_mats (e.g., ester oxygen, R-O(C=O)-R will only have one lone pair not two in this calculation)
    centers = [ i+np.ceil(min([ b[count,count] for b in bond_mats ])*0.5) for count,i in enumerate(sum(adj_mat)) ] # finds the number of charge centers bonded to each atom (determines hybridization) 
    s_char = np.array([ 1/(_+0.001) for _ in centers ]) # need s-character to assign positions of anions for precisely
    pol = np.array([ find_lewis.pol[_] for _ in elements_lower ]) # polarizability of each atom
    # Calculate final scores. For finding the preferred position of formal charges, some small corrections are made to the electronegativities of anion and cations based on neighboring atoms and hybridization.
    # The scores of ions are also adjusted by their ionization/reduction energy to provide a 0-baseline for all species regardless of charge state.
    rad_env = -np.sum(adj_mat*(0.1*pol/(100+pol)),axis=1)
    cat_en = en + rad_env
    an_en = en + np.sum(adj_mat*(0.1*en/(100+en)),axis=1) + 0.05*s_char
    scores = [ bmat_score(_,elements_lower,rings,cat_en,an_en,rad_env,e_tet,w_def=w_def,w_exp=w_exp,w_formal=w_formal,w_aro=w_aro,w_rad=w_rad,factor=factor,verbose=False) for _ in bond_mats ]
    # Sort by final scores
    bond_mats = [ _[1] for _ in sorted(zip(scores,bond_mats),key=lambda x:x[0]) ]
    # bond_mats = [ coordinate_covalent([i.lower() for i in elements], _[1]) for _ in sorted(zip(scores,bond_mats),key=lambda x:x[0]) ]
    scores = sorted(scores)
    tmp=deepcopy(bond_mats)
    return bond_mats,scores



# Returns a list with the number of electrons on each atom and a list with the number missing/surplus electrons on the atom
# 
# Inputs:  elements:  a list of element labels indexed to the adj_mat 
#          adj_mat:   np.array of atomic connections
#          bonding_pref: optional list of (index, bond_number) tuples that sets the target bond number of the indexed atoms
#          q_tot:     total charge on the molecule
#          fixed_bonds: optional list of (index_1,index_2,bond_number) tuples that creates fixed bonds between the index_1
#                       and index_2 atoms. No further bonds will be added or subtracted between these atoms.
#
# Optional inputs for ion and radical cases:
#          fc_0:      a list of formal charges on each atom
#          keep_lone: a list of atom index for which contains a radical 
#
# Returns: lone_electrons:
#          bonding_electrons:
#          core_electrons:
#          bond_mat:  an NxN matrix holding the bond orders between all atoms in the adj_mat
#          bonding_pref (optinal): optional list of (index, bond_number) tuples that sets the target bond number of the indexed atoms  
#
    
# put in old fine lewis algorithm because generate catalysis doesn't work with updated code
def old_find_lewis(elements,adj_mat_0,bonding_pref=[],q_tot=0,fixed_bonds=[],fc_0=None,keep_lone=[],return_pref=False,verbose=False,b_mat_only=False,return_FC=False,octet_opt=True,check_lewis_flag=False):
    
    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(find_lewis, "sat_dict"):

        find_lewis.lone_e = {'h':0, 'he':2,\
                             'li':0, 'be':2,                                                                                                                'b':0,     'c':0,     'n':2,     'o':4,     'f':6,    'ne':8,\
                             'na':0, 'mg':2,                                                                                                               'al':0,    'si':0,     'p':2,     's':4,    'cl':6,    'ar':8,\
                             'k':0, 'ca':2,  'sc':3, 'ti':4,  'v':5, 'cr':6, 'mn':7, 'fe':8, 'co':9, 'ni':10, 'cu':11, 'zn':None, 'ga':10,    'ge':0,    'as':3,    'se':4,    'br':6,    'kr':None,\
                             'rb':0, 'sr':2,  'y':3, 'zr':4, 'nb':5, 'mo':6, 'tc':7, 'ru':8, 'rh':9, 'pd':10, 'ag':None, 'cd':None, 'in':None, 'sn':None, 'sb':None, 'te':None,  'i':6,    'xe':None,\
                             'cs':0, 'ba':2, 'la':3, 'hf':4, 'ta':5,  'w':6, 're':7, 'os':8, 'ir':9, 'pt':10, 'au':None, 'hg':None, 'tl':None, 'pb':None, 'bi':None, 'po':None, 'at':None, 'rn':None }

        # Initialize periodic table
        find_lewis.periodic = {  "h": 1,  "he": 2,\
                                 "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                                 "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                  "k":19, "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                                 "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                                 "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}
        
        # Electronegativity ordering (for determining lewis structure)
        find_lewis.en = { "h" :2.3,  "he":4.16,\
                          "li":0.91, "be":1.58,                                                                                                               "b" :2.05, "c" :2.54, "n" :3.07, "o" :3.61, "f" :4.19, "ne":4.79,\
                          "na":0.87, "mg":1.29,                                                                                                               "al":1.61, "si":1.91, "p" :2.25, "s" :2.59, "cl":2.87, "ar":3.24,\
                          "k" :0.73, "ca":1.03, "sc":1.19, "ti":1.38, "v": 1.53, "cr":1.65, "mn":1.75, "fe":1.80, "co":1.84, "ni":1.88, "cu":1.85, "zn":1.59, "ga":1.76, "ge":1.99, "as":2.21, "se":2.42, "br":2.69, "kr":2.97,\
                          "rb":0.71, "sr":0.96, "y" :1.12, "zr":1.32, "nb":1.41, "mo":1.47, "tc":1.51, "ru":1.54, "rh":1.56, "pd":1.58, "ag":1.87, "cd":1.52, "in":1.66, "sn":1.82, "sb":1.98, "te":2.16, "i" :2.36, "xe":2.58,\
                          "cs":0.66, "ba":0.88, "la":1.09, "hf":1.16, "ta":1.34, "w" :1.47, "re":1.60, "os":1.65, "ir":1.68, "pt":1.72, "au":1.92, "hg":1.76, "tl":1.79, "pb":1.85, "bi":2.01, "po":2.19, "at":2.39, "rn":2.60} 

        # Polarizability ordering (for determining lewis structure)
        find_lewis.pol ={ "h" :4.5,  "he":1.38,\
                          "li":164.0, "be":377,                                                                                                               "b" :20.5, "c" :11.3, "n" :7.4, "o" :5.3,  "f" :3.74, "ne":2.66,\
                          "na":163.0, "mg":71.2,                                                                                                              "al":57.8, "si":37.3, "p" :25.0,"s" :19.4, "cl":14.6, "ar":11.1,\
                          "k" :290.0, "ca":161.0, "sc":97.0, "ti":100.0, "v": 87.0, "cr":83.0, "mn":68.0, "fe":62.0, "co":55, "ni":49, "cu":47.0, "zn":38.7,  "ga":50.0, "ge":40.0, "as":30.0,"se":29.0, "br":21.0, "kr":16.8,\
                          "rb":320.0, "sr":197.0, "y" :162,  "zr":112.0, "nb":98.0, "mo":87.0, "tc":79.0, "ru":72.0, "rh":66, "pd":26.1, "ag":55, "cd":46.0,  "in":65.0, "sn":53.0, "sb":43.0,"te":28.0, "i" :32.9, "xe":27.3,}

        # Bond energy dictionary {}-{}-{} refers to atom1, atom2 additional bonds number (1 refers to double bonds)
        # If energy for multiple bonds is missing, it means it's unusual to form multiple bonds, such value will be -10000.0, if energy for single bonds if missing, directly take multiple bonds energy as the difference 
        # From https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Bond_Energies
        #find_lewis.be = { "6-6-1": 267, "6-6-2":492, "6-7-1":310, "6-7-2":586, "6-8-1":387, "6-8-2":714, "7-8-1":406, "7-7-1":258, "7-7-2":781, "8-8-1":349, "8-16-1":523, "16-16-1":152}
        # Or from https://www2.chemistry.msu.edu/faculty/reusch/OrgPage/bndenrgy.htm ("6-15-1" is missing)
        # Remove 6-16-1:73
        find_lewis.be = { "6-6-1": 63, "6-6-2":117, "6-7-1":74, "6-7-2":140, "6-8-1":92.5, "6-8-2":172.5, "7-7-1":70.6, "7-7-2":187.6, "7-8-1":88, "8-8-1":84, "8-15-1":20, "8-16-1":6, "15-15-1":84,"15-15-2": 117, "15-16-1":70}
        
        # Initialize periodic table
        find_lewis.atomic_to_element = { find_lewis.periodic[i]:i for i in find_lewis.periodic.keys() }

    # Consistency check on fc_0 argument, if supplied
    if fc_0 is not None:

        if len(fc_0) != len(elements):
            print("ERROR in find_lewis: the fc_0 and elements lists must have the same dimensions.")
            quit()

        if int(sum(fc_0)) != int(q_tot):
            print("ERROR in find_lewis: the sum of formal charges does not equal q_tot.")
            quit()

    # Initalize elementa and atomic_number lists for use by the function
    atomic_number = [ find_lewis.periodic[i.lower()] for i in elements ]
    adj_mat = deepcopy(adj_mat_0)

    # identify ring atoms and number of rings
    rings=[]
    ring_size_list=range(11)[3:] # at most 10 ring structure
    
    for ring_size in ring_size_list:
        for j,Ej in enumerate(elements):
            is_ring,ring_ind = return_ring_atom(adj_mat,j,ring_size=ring_size)
            if is_ring and ring_ind not in rings:
                rings += [ring_ind]
    rings=[list(ring_inds) for ring_inds in rings]

    # Initially assign all valence electrons as lone electrons
    lone_electrons    = np.zeros(len(elements),dtype="int")    
    bonding_electrons = np.zeros(len(elements),dtype="int")    
    core_electrons    = np.zeros(len(elements),dtype="int")
    valence           = np.zeros(len(elements),dtype="int")
    bonding_target    = np.zeros(len(elements),dtype="int")
    valence_list      = np.zeros(len(elements),dtype="int")    
    
    for count_i,i in enumerate(elements):

        # Grab the total number of (expected) electrons from the atomic number
        N_tot = atomic_number[count_i]   

        # Determine the number of core/valence electrons based on row in the periodic table
        if N_tot > 86:
            print("ERROR in find_lewis: the algorithm isn't compatible with atomic numbers greater than 86 owing to a lack of rules for treating lanthanides. Exiting...")
            quit()

        elif N_tot > 54:
            N_tot -= 54
            core_electrons[count_i] = 54
            valence[count_i]        = 18

        elif N_tot > 36:
            N_tot -= 36
            core_electrons[count_i] = 36
            valence[count_i]        = 18

        elif N_tot > 18:
            N_tot -= 18
            core_electrons[count_i] = 18
            valence[count_i]        = 18

        elif N_tot > 10:
            N_tot -= 10
            core_electrons[count_i] = 10
            valence[count_i]        = 8

        elif N_tot > 2:
            N_tot -= 2
            core_electrons[count_i] = 2
            valence[count_i]        = 8

        lone_electrons[count_i] = N_tot
        valence_list[count_i] = N_tot

        # Assign target number of bonds for this atom
        if count_i in [ j[0] for j in bonding_pref ]:
            bonding_target[count_i] = next( j[1] for j in bonding_pref if j[0] == count_i )
        else:
            bonding_target[count_i] = N_tot - find_lewis.lone_e[elements[count_i].lower()]       

    # Loop over the adjmat and assign initial bonded electrons assuming single bonds (and adjust lone electrons accordingly)
    for count_i,i in enumerate(adj_mat_0):
        bonding_electrons[count_i] += sum(i)
        lone_electrons[count_i] -= sum(i)

    # Apply keep_lone: add one electron to such index    
    for count_i in keep_lone:
        lone_electrons[count_i] += 1
        
    # Eliminate all radicals by forming higher order bonds
    change_list = range(len(lone_electrons))
    bonds_made = []    
    loop_list   = [ (atomic_number[i],i) for i in range(len(lone_electrons)) ]
    loop_list   = [ i[1] for i in sorted(loop_list) ]

    # Check for special chemical groups
    for i in range(len(elements)):
        # Handle nitro groups
        if is_nitro(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ]
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],1)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            lone_electrons[O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[1],i] += 1

        # Handle sulfoxide groups
        if is_sulfoxide(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the thioketone atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind[0])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[O_ind[0],i] += 1

        # Handle sulfonyl groups
        if is_sulfonyl(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the sulfoxide atoms from the bonding_pref list
            bonding_pref += [(i,6)]
            bonding_pref += [(O_ind[0],2)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 2
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            bonds_made += [(i,O_ind[0])]
            bonds_made += [(i,O_ind[1])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[0],i] += 1
            adj_mat[O_ind[1],i] += 1            
        
        # Handle phosphate groups 
        if is_phosphate(i,adj_mat_0,elements) is True:
            O_ind      = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] in ["o","O"] ] # Index of single bonded O-P oxygens
            O_ind_term = [ j for j in O_ind if sum(adj_mat_0[j]) == 1 ] # Index of double bonded O-P oxygens
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the phosphate atoms from the bonding_pref list
            bonding_pref += [(i,5)]
            bonding_pref += [(O_ind_term[0],2)]  # during testing it ended up being important to only add a bonding_pref tuple for one of the terminal oxygens
            bonding_electrons[O_ind_term[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind_term[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind_term[0])]
            adj_mat[i,O_ind_term[0]] += 1
            adj_mat[O_ind_term[0],i] += 1

        # Handle cyano groups
        if is_cyano(i,adj_mat_0,elements) is True:
            C_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] in  ["c","C"] and sum(adj_mat_0[count_j]) == 2 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in C_ind ] # remove bonds involving the cyano atoms from the bonding_pref list
            bonding_pref += [(i,3)]
            bonding_pref += [(C_ind[0],4)]
            bonding_electrons[C_ind[0]] += 2
            bonding_electrons[i] += 2
            lone_electrons[C_ind[0]] -= 2
            lone_electrons[i] -= 2
            bonds_made += [(i,C_ind[0])]
            bonds_made += [(i,C_ind[0])]
            adj_mat[i,C_ind[0]] += 2
            adj_mat[C_ind[0],i] += 2

        # Handle isocyano groups
        if is_isocyano(i,adj_mat,elements) is True:
            C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in  ["c","C"] and sum(adj_mat[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in C_ind ] # remove bonds involving the cyano atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(C_ind[0],3)]
            bonding_electrons[C_ind[0]] += 2
            bonding_electrons[i] += 2
            lone_electrons[C_ind[0]] -= 2
            lone_electrons[i] -= 2
            bonds_made += [(i,C_ind[0])]
            bonds_made += [(i,C_ind[0])]
            adj_mat[i,C_ind[0]] += 2
            adj_mat[C_ind[0],i] += 2

    # Apply fixed_bonds argument
    off_limits=[]
    for i in fixed_bonds:

        # Initalize intermediate variables
        a = i[0]
        b = i[1]
        N = i[2]
        N_current = len([ j for j in bonds_made if (a,b) == j or (b,a) == j ]) + 1
        # Check that a bond exists between these atoms in the adjacency matrix
        if adj_mat_0[a,b] != 1:
            print("ERROR in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but the adjacency matrix doesn't reflect a bond. Exiting...")
            quit()

        # Check that less than or an equal number of bonds exist between these atoms than is requested
        if N_current > N:
            print("ERROR in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but {} bonds already exist between these atoms. There may be a conflict".format(N_current))
            print("                      between the special groups handling and the requested lewis_structure.")
            quit()

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[a] < (N - N_current):
            print("Warning in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(elements[a],lone_electrons[a]))

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[b] < (N - N_current):
            print("Warning in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(elements[b],lone_electrons[b]))
        
        # Make the bonds between the atoms
        for j in range(N-N_current):
            bonding_electrons[a] += 1
            bonding_electrons[b] += 1
            lone_electrons[a]    -= 1
            lone_electrons[b]    -= 1
            bonds_made += [ (a,b) ]

        # Append bond to off_limits group so that further bond additions/breaks do not occur.
        off_limits += [(a,b),(b,a)]

    # Turn the off_limits list into a set for rapid lookup
    off_limits = set(off_limits)
    
    # Adjust formal charges (if supplied)
    if fc_0 is not None:
        for count_i,i in enumerate(fc_0):
            if i > 0:
                #if lone_electrons[count_i] < i:
                    #print "ERROR in find_lewis: atom ({}, index {}) doesn't have enough lone electrons ({}) to be removed to satisfy the specified formal charge ({}).".format(elements[count_i],count_i,lone_electrons[count_i],i)
                    #quit()
                lone_electrons[count_i] = lone_electrons[count_i] - i
            if i < 0:
                lone_electrons[count_i] = lone_electrons[count_i] + int(abs(i))
        q_tot=0
    
    # diagnostic print            
    if verbose is True:
        print("Starting electronic structure:")
        print("\n{:40s} {:20} {:20} {:20} {:20} {}".format("elements","lone_electrons","bonding_electrons","core_electrons","formal_charge","bonded_atoms"))
        for count_i,i in enumerate(elements):
            print("{:40s} {:<20d} {:<20d} {:<20d} {:<20d} {}".format(elements[count_i],lone_electrons[count_i],bonding_electrons[count_i],core_electrons[count_i],\
                                                                     valence_list[count_i] - bonding_electrons[count_i] - lone_electrons[count_i],\
                                                                     ",".join([ "{}".format(count_j) for count_j,j in enumerate(adj_mat[count_i]) if j == 1 ])))

    # Initialize objects for use in the algorithm
    lewis_total = 1000
    lewis_lone_electrons = []
    lewis_bonding_electrons = []
    lewis_core_electrons = []
    lewis_valence = []
    lewis_bonding_target = []
    lewis_bonds_made = []
    lewis_adj_mat = []
    lewis_identical_mat = []
    
    # Determine the atoms with lone pairs that are unsatisfied as candidates for electron removal/addition to satisfy the total charge condition  
    happy = [ i[0] for i in bonding_pref if i[1] <= bonding_electrons[i[0]]]
    bonding_pref_ind = [i[0] for i in bonding_pref]
        
    # Determine is electrons need to be removed or added
    if q_tot > 0:
        adjust = -1
        octet_violate_e = []
        for count_j,j in enumerate(elements):
            if j.lower() in ["c","n","o","f","si","p","s","cl"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] > 8:
                    octet_violate_e += [count_j]
            elif j.lower() in ["br","i"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] > 18:
                    octet_violate_e += [count_j]
        
        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy and count_i not in octet_violate_e]
    
    elif q_tot < 0:
        adjust = 1
        octet_violate_e = []
        for count_j,j in enumerate(elements):
            if j.lower() in ["c","n","o","f","si","p","s","cl"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] < 8:
                    octet_violate_e += [count_j]
                    
            elif j.lower() in ["br","i"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] < 18:
                    octet_violate_e += [count_j]

        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy and count_i not in octet_violate_e]
        
    else:
        adjust = 1
        octet_violate_e = []
        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy ]
    
    # The outer loop checks each bonding structure produced by the inner loop for consistency with
    # the user specified "pref_bonding" and pref_argument with bonding electrons are
    for dummy_counter in range(lewis_total):

        lewis_loop_list = loop_list
        random.shuffle(lewis_loop_list)
        outer_counter     = 0
        inner_max_cycles  = 1000
        outer_max_cycles  = 1000
        bond_sat = False
        
        lewis_lone_electrons.append(deepcopy(lone_electrons))
        lewis_bonding_electrons.append(deepcopy(bonding_electrons))
        lewis_core_electrons.append(deepcopy(core_electrons))
        lewis_valence.append(deepcopy(valence))
        lewis_bonding_target.append(deepcopy(bonding_target))
        lewis_bonds_made.append(deepcopy(bonds_made))
        lewis_adj_mat.append(deepcopy(adj_mat))
        lewis_counter = len(lewis_lone_electrons) - 1
        
        # Adjust the number of electrons by removing or adding to the available lone pairs
        # The algorithm simply adds/removes from the first N lone pairs that are discovered
        random.shuffle(octet_violate_e)
        random.shuffle(normal_adjust)
        adjust_ind=octet_violate_e+normal_adjust
    
        if len(adjust_ind) >= abs(q_tot): 
            for i in range(abs(q_tot)):
                lewis_lone_electrons[-1][adjust_ind[i]] += adjust
                lewis_bonding_target[-1][adjust_ind[i]] += adjust 
        else:
            for i in range(abs(q_tot)):
                lewis_lone_electrons[-1][0] += adjust
                lewis_bonding_target[-1][0] += adjust

        # Search for an optimal lewis structure
        while bond_sat is False:
        
            # Initialize necessary objects
            change_list   = range(len(lewis_lone_electrons[lewis_counter]))
            inner_counter = 0
            bond_sat = True                
            # Inner loop forms bonds to remove radicals or underbonded atoms until no further
            # changes in the bonding pattern are observed.
            while len(change_list) > 0:
                change_list = []
                for i in lewis_loop_list:

                    # List of atoms that already have a satisfactory binding configuration.
                    happy = [ j[0] for j in bonding_pref if j[1] <= lewis_bonding_electrons[lewis_counter][j[0]]]            
                    
                    # If the current atom already has its target configuration then no further action is taken
                    if i in happy: continue

                    # If there are no lone electrons or too more bond formed then skip
                    if lewis_lone_electrons[lewis_counter][i] == 0: continue
                    
                    # Take action if this atom has a radical or an unsatifisied bonding condition
                    if lewis_lone_electrons[lewis_counter][i] % 2 != 0 or lewis_bonding_electrons[lewis_counter][i] != lewis_bonding_target[lewis_counter][i]:
                        # Try to form a bond with a neighboring radical (valence +1/-1 check ensures that no improper 5-bonded atoms are formed)
                        lewis_bonded_radicals = [ (-find_lewis.en[elements[count_j].lower()],count_j) for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and lewis_lone_electrons[lewis_counter][count_j] % 2 != 0 \
                                                  and 2*(lewis_bonding_electrons[lewis_counter][count_j]+1)+(lewis_lone_electrons[lewis_counter][count_j]-1) <= lewis_valence[lewis_counter][count_j]\
                                                  and lewis_lone_electrons[lewis_counter][count_j]-1 >= 0 and count_j not in happy ]

                        lewis_bonded_lonepairs= [ (-find_lewis.en[elements[count_j].lower()],count_j) for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and lewis_lone_electrons[lewis_counter][count_j] > 0 \
                                                  and 2*(lewis_bonding_electrons[lewis_counter][count_j]+1)+(lewis_lone_electrons[lewis_counter][count_j]-1) <= lewis_valence[lewis_counter][count_j] and lewis_lone_electrons[lewis_counter][count_j]-1 >= 0 \
                                                  and count_j not in happy ]
                        
                        # Sort by atomic number (cheap way of sorting carbon before other atoms, should probably switch over to electronegativities) 
                        lewis_bonded_radicals  = [ j[1] for j in sorted(lewis_bonded_radicals) ]
                        lewis_bonded_lonepairs = [ j[1] for j in sorted(lewis_bonded_lonepairs) ]
                        
                        # Correcting radicals is attempted first
                        if len(lewis_bonded_radicals) > 0:
                            lewis_bonding_electrons[lewis_counter][i] += 1
                            lewis_bonding_electrons[lewis_counter][lewis_bonded_radicals[0]] += 1
                            lewis_adj_mat[lewis_counter][i][lewis_bonded_radicals[0]] += 1
                            lewis_adj_mat[lewis_counter][lewis_bonded_radicals[0]][i] += 1 
                            lewis_lone_electrons[lewis_counter][i] -= 1
                            lewis_lone_electrons[lewis_counter][lewis_bonded_radicals[0]] -= 1
                            change_list += [i,lewis_bonded_radicals[0]]
                            lewis_bonds_made[lewis_counter] += [(i,lewis_bonded_radicals[0])]
                                                        
                        # Else try to form a bond with a neighboring atom with spare lone electrons (valence check ensures that no improper 5-bonded atoms are formed)
                        elif len(lewis_bonded_lonepairs) > 0:
                            lewis_bonding_electrons[lewis_counter][i] += 1
                            lewis_bonding_electrons[lewis_counter][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][i][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][lewis_bonded_lonepairs[0]][i] += 1
                            lewis_lone_electrons[lewis_counter][i] -= 1
                            lewis_lone_electrons[lewis_counter][lewis_bonded_lonepairs[0]] -= 1
                            change_list += [i,lewis_bonded_lonepairs[0]]
                            lewis_bonds_made[lewis_counter] += [(i,lewis_bonded_lonepairs[0])]
                            #lewis_bonds_en[lewis_counter] += 1.0/find_lewis.en[elements[i].lower()]/find_lewis.en[elements[lewis_bonded_lonepairs[0]].lower()]
                            
                # Increment the counter and break if the maximum number of attempts have been made
                inner_counter += 1
                if inner_counter >= inner_max_cycles:
                    print("WARNING: maximum attempts to establish a reasonable lewis-structure exceeded ({}).".format(inner_max_cycles))
            
            # Check if the user specified preferred bond order has been achieved.
            if bonding_pref is not None:
                unhappy = [ i[0] for i in bonding_pref if i[1] != lewis_bonding_electrons[lewis_counter][i[0]]]            
                if len(unhappy) > 0:

                    # Break the first bond involving one of the atoms bonded to the under/over coordinated atoms
                    ind = set([unhappy[0]] + [ count_i for count_i,i in enumerate(adj_mat_0[unhappy[0]]) if i == 1 and (count_i,unhappy[0]) not in off_limits ])
                    
                    # Check if a rearrangment is possible, break if none are available
                    try:
                        break_bond = next( i for i in lewis_bonds_made[lewis_counter] if i[0] in ind or i[1] in ind )
                    except:
                        print("WARNING: no further bond rearrangments are possible and bonding_pref is still not satisfied.")
                        break
                    
                    # Perform bond rearrangment
                    lewis_bonding_electrons[lewis_counter][break_bond[0]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[0]] += 1
                    lewis_adj_mat[lewis_counter][break_bond[0]][break_bond[1]] -= 1
                    lewis_adj_mat[lewis_counter][break_bond[1]][break_bond[0]] -= 1
                    lewis_bonding_electrons[lewis_counter][break_bond[1]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[1]] += 1

                    # Remove the bond from the list and reorder lewis_loop_list so that the indices involved in the bond are put last                
                    lewis_bonds_made[lewis_counter].remove(break_bond)
                    lewis_loop_list.remove(break_bond[0])
                    lewis_loop_list.remove(break_bond[1])
                    lewis_loop_list += [break_bond[0],break_bond[1]]
                                        
                    # Update the bond_sat flag
                    bond_sat = False
                    
                # Increment the counter and break if the maximum number of attempts have been made
                outer_counter += 1
                    
                # Periodically reorder the list to avoid some cyclical walks
                if outer_counter % 100 == 0:
                    lewis_loop_list = reorder_list(lewis_loop_list,atomic_number)

                # Print diagnostic upon failure
                if outer_counter >= outer_max_cycles:
                    print("WARNING: maximum attempts to establish a lewis-structure consistent")
                    print("         with the user supplied bonding preference has been exceeded ({}).".format(outer_max_cycles))
                    break
        
        # Re-apply keep_lone: remove one electron from such index    
        for count_i in keep_lone:
            lewis_lone_electrons[lewis_counter][count_i] -= 1

        # Special cases, share pair of electrons
        total_electron=np.array(lewis_lone_electrons[lewis_counter])+np.array(lewis_bonding_electrons[lewis_counter])*2
        
        # count for atom which doesn't satisfy
        # Notice: need systematical check for this part !!!
        unsatisfy = [count_t for count_t,te in enumerate(total_electron) if te > 2 and te < 8 and te % 2 ==0]
        for uns in unsatisfy:
            full_connect=[count_i for count_i,i in enumerate(adj_mat_0[uns]) if i == 1 and total_electron[count_i] == 8 and lewis_lone_electrons[lewis_counter][count_i] >= 2]
            if len(full_connect) > 0:

                lewis_lone_electrons[lewis_counter][full_connect[0]]-=2
                lewis_bonding_electrons[lewis_counter][uns]+=1
                lewis_bonding_electrons[lewis_counter][full_connect[0]]+=1
                lewis_adj_mat[lewis_counter][uns][full_connect[0]]+=1
                lewis_adj_mat[lewis_counter][full_connect[0]][uns]+=1 

        # Delete last entry in the lewis np.arrays if the electronic structure is not unique: introduce identical_mat includes both info of bond_mats and formal_charges
        identical_mat=np.vstack([lewis_adj_mat[-1], np.array([ valence_list[k] - lewis_bonding_electrons[-1][k] - lewis_lone_electrons[-1][k] for k in range(len(elements)) ]) ])
        lewis_identical_mat.append(identical_mat)

        if array_unique(lewis_identical_mat[-1],lewis_identical_mat[:-1])[0] is False:
            lewis_lone_electrons    = lewis_lone_electrons[:-1]
            lewis_bonding_electrons = lewis_bonding_electrons[:-1]
            lewis_core_electrons    = lewis_core_electrons[:-1]
            lewis_valence           = lewis_valence[:-1]
            lewis_bonding_target    = lewis_bonding_target[:-1]
            lewis_bonds_made        = lewis_bonds_made[:-1]
            lewis_adj_mat           = lewis_adj_mat[:-1]
            lewis_identical_mat     = lewis_identical_mat[:-1]
            
    # Find the total number of lone electrons in each structure
    lone_electrons_sums = []
    for i in range(len(lewis_lone_electrons)):
        lone_electrons_sums.append(sum(lewis_lone_electrons[i]))
        
    # Find octet violations in each structure
    octet_violations = []
    for i in range(len(lewis_lone_electrons)):
        ov = 0
        if octet_opt is True:
            for count_j,j in enumerate(elements):
                if j.lower() in ["c","n","o","f","si","p","s","cl","br","i"] and count_j not in bonding_pref_ind:
                    if lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j] != 8 and lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j] != 18:
                        ov += 1
        octet_violations.append(ov)

    ## Calculate bonding energy
    lewis_bonds_energy = []
    for bonds_made in lewis_bonds_made:
        for lb,bond_made in enumerate(bonds_made): bonds_made[lb]=tuple(sorted(bond_made))
        count_bonds_made = ["{}-{}-{}".format(min(atomic_number[bm[0]],atomic_number[bm[1]]),max(atomic_number[bm[0]],atomic_number[bm[1]]),bonds_made.count(bm) ) for bm in set(bonds_made)]
        lewis_bonds_energy += [sum([find_lewis.be[cbm] if cbm in find_lewis.be.keys() else -10000.0 for cbm in count_bonds_made  ]) ]

    # normalize the effect
    lewis_bonds_energy = [-be/max(1,max(lewis_bonds_energy)) for be in lewis_bonds_energy]

    ## Find the total formal charge for each structure
    formal_charges_sums = []
    for i in range(len(lewis_lone_electrons)):
        fc = 0
        for j in range(len(elements)):
            fc += abs(valence_list[j] - lewis_bonding_electrons[i][j] - lewis_lone_electrons[i][j])
        formal_charges_sums.append(fc)
    
    ## Find formal charge eletronegativity contribution
    lewis_formal_charge = [ [ valence_list[i] - lewis_bonding_electrons[_][i] - lewis_lone_electrons[_][i] for i in range(len(elements)) ] for _ in range(len(lewis_lone_electrons)) ]
    lewis_keep_lone     = [ [ count_i for count_i,i in enumerate(lone) if i % 2 != 0] for lone in lewis_lone_electrons]
    lewis_fc_aro = [] # form aromatic rings to stablize charge/radical 
    lewis_fc_en  = [] # Electronegativity for stabling charge/radical
    lewis_fc_pol = [] # Polarizability for stabling charge/radical
    lewis_fc_hc  = [] # Hyper-conjugation contribution

    for i in range(len(lewis_lone_electrons)):

        formal_charge = lewis_formal_charge[i]
        radical_atom = lewis_keep_lone[i]
        fc_ind = [(count_j,j) for count_j,j in enumerate(formal_charge) if j != 0]

        for R_ind in radical_atom:  # assign +0.5 for radical
            fc_ind += [(R_ind,0.5)]
        
        # initialize en,pol and hc
        fc_aro,fc_en,fc_pol,fc_hc = 0,0,0,0

        if len(rings) > 0 and len(fc_ind) > 0:

            # loop over all rings
            for ring in rings:

                # if radical/charged atoms in this ring, and this ring consists of even number atoms
                #if len(ring) % 2 == 0 and (True in [ind[0] in ring for ind in fc_ind]):
                if len(ring) % 2 == 0:
        
                    bond_madesi = lewis_bonds_made[i]
                    bond_madesi_ring = [bmade for bmade in bond_madesi if (bmade[0] in ring and bmade[1] in ring) ]

                    # check whether generate an aromatic ring
                    if len(bond_madesi_ring) == len(ring) / 2 and len(set(sum([list(bond) for bond in bond_madesi_ring],[]))) == len(ring): fc_aro += -1

        # Loop over formal charges and radicals
        for count_fc in fc_ind:

            ind = count_fc[0]    
            charge = count_fc[1]
            # Count the self contribution: (-) on the most electronegative atom and (+) on the least electronegative atom
            fc_en += 10 * charge * find_lewis.en[elements[ind].lower()]
             
            # Find the nearest and next-nearest atoms for each formal_charge/radical contained atom
            gs = graph_seps(adj_mat_0)
            nearest_atoms = [count_k for count_k,k in enumerate(lewis_adj_mat[i][ind]) if k >= 1] 
            NN_atoms = list(set([ count_j for count_j,j in enumerate(gs[ind]) if j == 2 ]))
            
            # only count when en > en(C)
            fc_en += charge*(sum([find_lewis.en[elements[count_k].lower()] for count_k in nearest_atoms if find_lewis.en[elements[count_k].lower()] > 2.54] )+\
                             sum([find_lewis.en[elements[count_k].lower()] for count_k in NN_atoms if find_lewis.en[elements[count_k].lower()] > 2.54] ) * 0.1 )

            if charge < 0: # Polarizability only affects negative charge
                fc_pol += charge*sum([find_lewis.pol[elements[count_k].lower()] for count_k in nearest_atoms ])
            
            # find hyper-conjugation strcuture
            nearby_carbon = [nind for nind in nearest_atoms if elements[nind].lower()=='c']
            for carbon_ind in nearby_carbon:
                carbon_nearby=[nind for nind in NN_atoms if lewis_adj_mat[i][carbon_ind][nind] >= 1 and elements[nind].lower() in ['c','h']]
                if len(carbon_nearby) == 3: fc_hc -= charge*(len([nind for nind in carbon_nearby if elements[nind].lower() == 'c'])*2 + len([nind for nind in carbon_nearby if elements[nind].lower() == 'h']))

        lewis_fc_aro.append(fc_aro)
        lewis_fc_en.append(fc_en)        
        lewis_fc_pol.append(fc_pol)        
        lewis_fc_hc.append(fc_hc)        

    # normalize the effect
    lewis_fc_en = [lfc/max(1,max(abs(np.array(lewis_fc_en)))) for lfc in lewis_fc_en]
    lewis_fc_pol= [lfp/max(1,max(abs(np.array(lewis_fc_pol)))) for lfp in lewis_fc_pol]
    
    # Add the total number of radicals to the total formal charge to determine the criteria.
    # The radical count is scaled by 0.01 and the lone pair count is scaled by 0.001. This results
    # in the structure with the lowest formal charge always being returned, and the radical count 
    # only being considered if structures with equivalent formal charges are found, and likewise with
    # the lone pair count. The structure(s) with the lowest score will be returned.
    lewis_criteria = []
    for i in range(len(lewis_lone_electrons)):
        #lewis_criteria.append( 10.0*octet_violations[i] + abs(formal_charges_sums[i]) + 0.1*sum([ 1 for j in lewis_lone_electrons[i] if j % 2 != 0 ]) + 0.001*lewis_bonds_energy[i]  + 0.00001*lewis_fc_en[i] + 0.000001*lewis_fc_pol[i] + 0.0000001*lewis_fc_hc[i]) 
        lewis_criteria.append( 10.0 * octet_violations[i] + abs(formal_charges_sums[i]) + 0.1 * sum([ 1 for j in lewis_lone_electrons[i] if j % 2 != 0 ]) + 0.05 * lewis_fc_aro[i] + 0.01 * lewis_fc_en[i] + \
                               0.005 * lewis_fc_pol[i] + 0.0001 * lewis_fc_hc[i] + 0.0001 * lewis_bonds_energy[i])
    #print(lewis_criteria)
    #print(lewis_bonds_made)
    #exit()
    best_lewis = [i[0] for i in sorted(enumerate(lewis_criteria), key=lambda x:x[1])]  # sort from least to most and return a list containing the origial list's indices in the correct order
    best_lewis = [ i for i in best_lewis if lewis_criteria[i] == lewis_criteria[best_lewis[0]] ]    
    
    # Finally check formal charge to keep those with 
    lewis_re_fc     = [ lewis_formal_charge[_]+lewis_keep_lone[_] for _ in best_lewis]
    appear_times    = [ lewis_re_fc.count(i) for i in lewis_re_fc]
    best_lewis      = [best_lewis[i] for i in range(len(lewis_re_fc)) if appear_times[i] == max(appear_times) ] 
    
    # Apply keep_lone information, remove the electron to form lone electron
    for i in best_lewis:
        for j in keep_lone:
            lewis_lone_electrons[i][j] -= 1

    # Print diagnostics
    if verbose is True:
        for i in best_lewis:
            print("Bonding Matrix  {}".format(i))
            print("Formal_charge:  {}".format(formal_charges_sums[i]))
            print("Lewis_criteria: {}\n".format(lewis_criteria[i]))
            print("{:<40s} {:<40s} {:<15s} {:<15s}".format("Elements","Bond_Mat","Lone_Electrons","FC"))
            for j in range(len(elements)):
                print("{:<40s} {}    {} {}".format(elements[j]," ".join([ str(k) for k in lewis_adj_mat[i][j] ]),lewis_lone_electrons[i][j],valence_list[j] - lewis_bonding_electrons[i][j] - lewis_lone_electrons[i][j]))
            print (" ")

    # If only the bonding matrix is requested, then only that is returned
    if b_mat_only is True:
        if return_FC is False:  
            return [ lewis_adj_mat[_] for _ in best_lewis ]
        else:
            return [ lewis_adj_mat[_] for _ in best_lewis ], [ lewis_formal_charge[_] for _ in best_lewis ]

    # return like check_lewis function
    if check_lewis_flag is True:
        if return_pref is True:
            return lewis_lone_electrons[best_lewis[0]], lewis_bonding_electrons[best_lewis[0]], lewis_core_electrons[best_lewis[0]],bonding_pref
        else:
            return lewis_lone_electrons[best_lewis[0]], lewis_bonding_electrons[best_lewis[0]], lewis_core_electrons[best_lewis[0]]
    
    # Optional bonding pref return to handle cases with special groups
    if return_pref is True:
        if return_FC is False:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],bonding_pref
        else:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],[ lewis_formal_charge[_] for _ in best_lewis ],bonding_pref 

    else:
        if return_FC is False:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ]
        else:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],[ lewis_formal_charge[_] for _ in best_lewis ]


class LewisStructureError(Exception):

    def __init__(self, message="An error occured in a find_lewis() call."):
        self.message = message
        super().__init__(self.message)

def coordinate_covalent(elements, bond_mat):
    covalent_flag=True
    while covalent_flag:
        e=return_e(bond_mat)
        covalent_flag=False
        for _, mat in enumerate(bond_mat):
            formal=return_formals(bond_mat, elements)
            en_diff=[]
            connect_index=[]
            if formal[_]>0 and (e[_]<find_lewis.n_electrons[elements[_]] or find_lewis.expand_octet[elements[_]] is True): # cation that could accept electrons (means that NH4+ would be removed)
                for count_j, j in enumerate(mat):
                    if j and (formal[count_j]<0 and bond_mat[count_j, count_j] > 1): # a connected anion with a paired electron that could have a coordinate covalent bond
                        # find the difference in electronegativity
                        connect_index.append(count_j)
                        en_diff.append(abs(find_lewis.en[elements[_]]-find_lewis.en[elements[count_j]]))
                        covalent_flag=True
                if covalent_flag==True and len(en_diff)>0:
                    en_diff=np.array(en_diff)
                    ind=connect_index[int(np.where(en_diff==max(en_diff))[0][0])]
                    bond_mat[ind, ind]=bond_mat[ind, ind]-2
                    bond_mat[_, ind]=bond_mat[_, ind]+1
                    bond_mat[ind, _]=bond_mat[_, ind]
                    formal[ind]=formal[ind]+1
                    formal[_]=formal[_]-1
    formal=return_formals(bond_mat, elements)
    return bond_mat  
# Helper function for iterating over the placement of charges
def gen_init(obj_fun,adj_mat,elements,rings,q):

    # Array of atom-wise electroneutral electron expectations for convenience.
    eneutral = np.array([ find_lewis.valence[_] for _ in elements ]) 
    #print('eneutral in gen init: ', eneutral)   

    # Array of atom-wise octet requirements for determining electron deficiencies
    e_tet = np.array([ find_lewis.n_electrons[_] for _ in elements ])
    
    # Initial neutral bond electron matrix with sigma bonds in place
    bond_mat = deepcopy(adj_mat) + np.diag(np.array([ _ - sum(adj_mat[count]) for count,_ in enumerate(eneutral) ]))
    #print('bond mat in gen init: ', bond_mat)
    # Correct atoms with negative charge using q (if anions)
    qeff = q 
    # Generate the graph matrix for bi-molecular system
    gs=graph_seps(adj_mat)       
    #print('qeff: ', qeff)
    n_ind = [ _ for _ in range(len(bond_mat)) if bond_mat[_,_] < 0 ]
    while (len(n_ind)>0 and qeff<0):
        bond_mat[n_ind[0],n_ind[0]] += 1
        qeff += 1
        n_ind = [ _ for _ in range(len(bond_mat)) if bond_mat[_,_] < 0 ]        
    
    # Correct atoms with negative charge using lone electrons
    n_ind = [ _ for _ in range(len(bond_mat)) if bond_mat[_,_] < 0 ]
    l_ind = [ _ for _ in range(len(bond_mat)) if bond_mat[_,_] > 0 ] 
    while (len(n_ind)>0 and len(l_ind)>0):
        for i in l_ind:
            try:
                def_atom = n_ind.pop(0)                
                bond_mat[def_atom,def_atom] += 1
                bond_mat[i,i] -= 1
            except:
                continue
        n_ind = [ _ for _ in range(len(bond_mat)) if bond_mat[_,_] < 0 ]
        l_ind = [ _ for _ in range(len(bond_mat)) if bond_mat[_,_] > 0 ] 
    # Raise error if there are still negative charges on the diagonal
    if len([ _ for _ in range(len(bond_mat)) if bond_mat[_,_] < 0 ]):
        raise LewisStructureError("Incompatible charge state and adjacency matrix.")
    # Correct expanded octets if possible (while performs CT from atoms with expanded octets
    # to deficient atoms until there are no more expanded octets or no more deficient atoms)
    e_ind = [ count for count,_ in enumerate(return_expanded(bond_mat,elements,e_tet)) if _ > 0 and bond_mat[count,count] > 0 ]
    d_ind = [ count for count,_ in enumerate(return_def(bond_mat,elements,e_tet)) if _ < 0 ]    
    graph_flag=True
    while (len(e_ind)>0 and len(d_ind)>0 and graph_flag):
        for i in e_ind:
            try:
                for j in d_ind:
                    if gs[j, i] != -1:
                        def_atom=j
                        break
                if gs[def_atom, i] != -1:
                    bond_mat[def_atom,def_atom] += 1
                    bond_mat[i,i] -= 1
            except:
                continue
        e_ind = [ count for count,_ in enumerate(return_expanded(bond_mat,elements,e_tet)) if _ > 0 and bond_mat[count,count] > 0 ]
        d_ind = [ count for count,_ in enumerate(return_def(bond_mat,elements,e_tet)) if _ < 0 ]
        graph_flag=False
        for _ in e_ind:
            for i in d_ind:
                if gs[_, i] != -1: graph_flag=True
    # Get the indices of atoms in rings < 10 (used to determine if multiple double bonds and alkynes are allowed on an atom)
    ring_atoms = { j for i in [ _ for _ in rings if len(_) < 10 ] for j in i }
    # bond_mat=coordinate_covalent(elements, bond_mat)
#    print("gen_init")
#    print(elements)
#    print(bond_mat)
#    print(return_formals(bond_mat, elements))
    # If charge is being added, then add to the most electronegative atoms first
    if qeff<0:

        # Non-hydrogen atoms
        heavies = [ count for count,_ in enumerate(elements) ] # there was a condition inside: if _ != "h"

        heav = list(itertools.combinations_with_replacement(heavies, int(abs(qeff))))
        # Loop over all q-combinations of heavy atoms
        for i in itertools.combinations_with_replacement(heavies, int(abs(qeff))):

            # Create a fresh copy of the initial be_mat and add charges
            tmp = copy(bond_mat)
            for _ in i: tmp[_,_] += 1
            # Find reactive atoms (i.e., atoms with unbound electron(s) or deficient atoms or a formal charge)
            e = return_e(tmp)
            f = return_formals(tmp,elements)
            reactive = [ count for count,_ in enumerate(elements) if ( tmp[count,count] or e[count] < find_lewis.n_electrons[_] or f[count] != 0 ) ]
            
            # Form bonded structure
            for j in reactive:
                while valid_bonds(j,tmp,elements,reactive,ring_atoms):            
                    for k in valid_bonds(j,tmp,elements,reactive,ring_atoms): tmp[k[1],k[2]]+=k[0]
#            print('---------')
#            print(tmp)
#            print(return_formals(tmp, elements))
#            print('---------')
            yield obj_fun(tmp),tmp, reactive

    # If charge is being removed, then remove from the least electronegative atoms first
    elif qeff>0:
        # Atoms with unbound electrons
        lonelies = [ count for count,_ in enumerate(bond_mat) if bond_mat[count,count] > 0 ]
        #print(bond_mat)
        # Loop over all q-combinations of atoms with unbound electrons to be oxidized
        for i in itertools.combinations_with_replacement(lonelies, qeff):

            # This construction is used to handle cases with q>1 to avoid taking more electrons than are available.
            tmp = copy(bond_mat)
            e = return_e(tmp)
            f = return_formals(tmp,elements)
            reactive = [ count for count,_ in enumerate(elements) if ( tmp[count,count] or e[count] < find_lewis.n_electrons[_] or f[count] != 0 ) ]
            # Form bonded structure
            for j in reactive:
                while valid_bonds(j,tmp,elements,reactive,ring_atoms):            
                    for k in valid_bonds(j,tmp,elements,reactive,ring_atoms): tmp[k[1],k[2]]+=k[0]

            flag = True
            for j in i:
                if tmp[j,j] > 0:
                    tmp[j,j] -= 1
                else:
                    flag = False
            if not flag:
                continue

            '''
            # Find reactive atoms (i.e., atoms with unbound electron(s) or deficient atoms or a formal charge)
            e = return_e(tmp)
            f = return_formals(tmp,elements)
            reactive = [ count for count,_ in enumerate(elements) if ( tmp[count,count] or e[count] < find_lewis.n_electrons[_] or f[count] != 0 ) ]
            # Form bonded structure
            for j in reactive:
                while valid_bonds(j,tmp,elements,reactive,ring_atoms):            
                    for k in valid_bonds(j,tmp,elements,reactive,ring_atoms): tmp[k[1],k[2]]+=k[0]
            '''
            yield obj_fun(tmp),tmp,reactive
        
    else:
        # Find reactive atoms (i.e., atoms with unbound electron(s) or deficient atoms or a formal charge)
        e = return_e(bond_mat)
        f = return_formals(bond_mat,elements)
        reactive = [ count for count,_ in enumerate(elements) if ( bond_mat[count,count] or e[count] < find_lewis.n_electrons[_] or f[count] != 0 ) ]
        # Form bonded structure
        for j in reactive:
            while valid_bonds(j,bond_mat,elements,reactive,ring_atoms):            
                for k in valid_bonds(j,bond_mat,elements,reactive,ring_atoms): bond_mat[k[1],k[2]]+=k[0]
        yield obj_fun(bond_mat),bond_mat,reactive

# recursive function that generates the lewis structures
# iterates over all valid moves, performs the move, and recursively calls itself with the updated bond matrix    
def gen_all_lstructs(obj_fun, bond_mats, scores, elements, reactive, rings, ring_atoms, bridgeheads, seps, min_score, ind=0, counter=0, N_score=100, N_max=10000, min_opt=False, min_win=False):

    # Loop over all possible moves, recursively calling this function to account for the order dependence. 
    # This could get very expensive very quickly, but with a well-curated moveset things are still very quick for most tested chemistries. 
    for j in valid_moves(bond_mats[ind],elements,reactive,rings,ring_atoms,bridgeheads,seps):
        # Carry out moves on trial bond_mat
        tmp = copy(bond_mats[ind])        
        for k in j: tmp[k[1],k[2]]+=k[0]

        # Check that the resulting bond_mat is not already in the existing bond_mats
        if bmat_unique(tmp,bond_mats):
            bond_mats += [tmp]
            scores += [obj_fun(tmp)]

            # Check if a new best Lewis structure has been found, if so, then reset counter and record new best score
            if scores[-1] <= min_score:
                counter = 0
                min_score = scores[-1]
            else:
                counter += 1

            # Break if too long (> N_score) has passed without finding a better Lewis structure
            if counter >= N_score:
                return bond_mats,scores,min_score,counter

            # If min_opt=True then the search is run in a greedy mode where only moves that reduce the score are accepted
            if min_opt:
                if counter == 0:
                    # Recursively call this function with the updated bond_mat resulting from this iteration's move. 
                    bond_mats,scores,min_score,counter = gen_all_lstructs(obj_fun,bond_mats,scores,elements,reactive,rings,ring_atoms,bridgeheads,seps,\
                                                                          min_score,ind=len(bond_mats)-1,counter=counter,N_score=N_score,N_max=N_max,min_opt=min_opt,min_win=min_win)
                
            else:
                # min_win option allows the search to follow structures that increase the score up to min_win above the score of the best structure
                if min_win:
                    if (scores[-1]-min_score) < min_win:
                        # Recursively call this function with the updated bond_mat resulting from this iteration's move. 
                        bond_mats,scores,min_score,counter = gen_all_lstructs(obj_fun,bond_mats,scores,elements,reactive,rings,ring_atoms,bridgeheads,seps,\
                                                                              min_score,ind=len(bond_mats)-1,counter=counter,N_score=N_score,N_max=N_max,min_opt=min_opt,min_win=min_win)

                # otherwise all structures are recursively explored (can be very expensive)
                else:
                    # Recursively call this function with the updated bond_mat resulting from this iteration's move. 
                    bond_mats,scores,min_score,counter = gen_all_lstructs(obj_fun,bond_mats,scores,elements,reactive,rings,ring_atoms,bridgeheads,seps,\
                                                                          min_score,ind=len(bond_mats)-1,counter=counter,N_score=N_score,N_max=N_max,min_opt=min_opt,min_win=min_win)
                    
        # Break if max has been encountered.
        if len(bond_mats) > N_max:
            return bond_mats,scores,min_score,counter
    
    return bond_mats,scores,min_score,counter


    
# Helper function for gen_all_lstructs that checks if a proposed bond_mat has already been encountered
def bmat_unique(new_bond_mat,old_bond_mats):
    for i in old_bond_mats:
        if all_zeros(i-new_bond_mat):
            return False
    return True 

# Helper function for bmat_unique that checks is a numpy array is all zeroes (uses short-circuit logic to speed things up in contrast to np.any)
def all_zeros(m):
    for _ in m.flat:
        if _:
            return False # short-circuit logic at first non-zero
    return True

# Helper function for gen_all_lstructs that yields all valid moves for a given bond_mat
# Returns a list of tuples of the form (#,i,j) where i,j are the indices in bond_mat
# and # is the value to be added to that position.
def valid_moves(bond_mat,elements,reactive,rings,ring_atoms,bridgeheads,seps):

    e = return_e(bond_mat) # current number of electrons associated with each atom

    # Loop over the individual atoms and determine the moves that apply
    for i in reactive:

        # All of these moves involve forming a double bond with the i atom. Constraints that are common to all of the moves are checked here.
        # These are avoiding forming alkynes/allenes in rings and Bredt's rule (forming double-bonds at bridgeheads)
        if i not in bridgeheads and ( i not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[i]) if count != i and _ > 1 ]) == 0 ):

            # Move 1: i is electron deficient and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is turned into a new i-j pi-bond.
            if e[i]+2 <= find_lewis.n_electrons[elements[i]] or find_lewis.expand_octet[elements[i]]:
                for j in return_connections(i,bond_mat,inds=reactive):
                    for k in [ _ for _ in return_connections(j,bond_mat,inds=reactive,min_order=2) if _ != i ]:
                        yield [(1,i,j),(1,j,i),(-1,j,k),(-1,k,j)]

            # Move 2: i has a radical and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is homolytically broken and a new pi-bond is formed between i and j
            if bond_mat[i,i] % 2 != 0 and e[i] < find_lewis.n_electrons[elements[i]]:
                for j in return_connections(i,bond_mat,inds=reactive):
                    for k in [ _ for _ in return_connections(j,bond_mat,inds=reactive,min_order=2) if _ != i ]:
                        yield [(1,i,j),(1,j,i),(-1,j,k),(-1,k,j),(-1,i,i),(1,k,k)]

            # Move 3: i has a lone pair and has an adjacent pi-bond between neighbor and next-nearest neighbor atoms, j and k, then the j-k pi-bond is heterolytically broken to form a lone pair on k and a new pi-bond is formed between i and j
            if bond_mat[i,i] >= 2:
                for j in return_connections(i,bond_mat,inds=reactive):
                    for k in [ _ for _ in return_connections(j,bond_mat,inds=reactive,min_order=2) if _ != i ]:
                        yield [(1,i,j),(1,j,i),(-1,j,k),(-1,k,j),(-2,i,i),(2,k,k)]

            # Move 4: i has a radical and a neighbor with unbound electrons
            if bond_mat[i,i] % 2 != 0 and ( find_lewis.expand_octet[elements[i]] or e[i] < find_lewis.n_electrons[elements[i]] ):

                # Check on connected atoms
                for j in return_connections(i,bond_mat,inds=reactive):

                    # Electron available @j
                    if bond_mat[j,j] > 0:

                        # Straightforward homogeneous bond formation if j is deficient or can expand octet
                        if ( find_lewis.expand_octet[elements[j]] or e[j] < find_lewis.n_electrons[elements[j]] ):

                            # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                            if j not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[j]) if count != j and _ > 1 ]) == 0:                  
                                yield [(1,i,j),(1,j,i),(-1,i,i),(-1,j,j)]

                        # Check if CT from j can be performed to an electron deficient atom or one that can expand its octet. This move is necessary for
                        # This moved used to be performed as an else to the previous statement, but would miss some ylides. Now it is run in all cases to be safer.                                          
                        if bond_mat[j,j] > 1:
                            for k in reactive:
                                if k != i and k != j and ( find_lewis.expand_octet[elements[k]] or e[k] < find_lewis.n_electrons[elements[k]] ):

                                    # Check that ring constraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                                    if j not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[j]) if count != j and _ > 1 ]) == 0:                  
                                        yield [(1,i,j),(1,j,i),(-1,i,i),(-2,j,j),(1,k,k)]
                                                    
            # Move 5: i has a lone pair and a neighbor capable of forming a double bond, then a new pi-bond is formed with the neighbor from the lone pair
            if bond_mat[i,i] >= 2:
                for j in return_connections(i,bond_mat,inds=reactive):
                    # Check ring conditions on j
                    if j not in bridgeheads and ( j not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[j]) if count != j and _ > 1 ]) == 0 ):
                        # Check octet conditions on j
                        if find_lewis.expand_octet[elements[j]] or e[j]+2 <= find_lewis.n_electrons[elements[j]]:                    
                            yield [(1,i,j),(1,j,i),(-2,i,i)]
                            
        # Move 6: i has a pi bond with j and the electronegativity of i is >= j, or a favorable change in aromaticity occurs, then the pi-bond is turned into a lone pair on i
        for j in return_connections(i,bond_mat,inds=reactive,min_order=2):
            if find_lewis.en[elements[i]] > find_lewis.en[elements[j]] or delta_aromatic(bond_mat,rings,move=((-1,i,j),(-1,j,i),(2,i,i))) or e[j] > find_lewis.n_electrons[elements[i]]:
                yield [(-1,i,j),(-1,j,i),(2,i,i)]

        # Move 7: i is electron deficient, bonded to j with unbound electrons, and the electronegativity of i is >= j, then an electron is tranferred from j to i
                # Note: very similar to move 4 except that a double bond is not formed. This is sometimes needed when j cannot expand its octet (as required by bond formation) but i still needs a full octet.
        if e[i] < find_lewis.n_electrons[elements[i]]:
            for j in return_connections(i,bond_mat,inds=reactive):
                if bond_mat[j,j] > 0 and find_lewis.en[elements[i]] > find_lewis.en[elements[j]]:
                    yield [(-1,j,j),(1,i,i)]

        # Move 8: i has an expanded octet and unbound electrons, then charge transfer to an atom within three bonds (controlled by local option) that is electron deficient or can expand its octet is attempted.
        if e[i] > find_lewis.n_electrons[elements[i]] and bond_mat[i,i] > 0:
            for j in reactive:
                if j != i and seps[i,j] < 3 and ( find_lewis.expand_octet[elements[j]] or e[j] < find_lewis.n_electrons[elements[j]] ):
                    yield [(-1,i,i),(1,j,j)]

        # # Move 9: i is deficient then charge transfer from an atom within three bonds is attempted (not clear if this is necessary anymore)
        # if e[i] < find_lewis.n_electrons[elements[i]]:
        #     for j in reactive:
        #         if j != i and seps[i,j] < 3 and bond_mat[j,j] > 0:
        #             yield [(1,i,i),(-1,j,j)]

    # Move 9: shuffle aromatic and anti-aromatic bonds 
    for i in rings:
        if is_aromatic(bond_mat,i) and len(i) % 2 == 0: 

            # Find starting point
            loop_ind = None
            for count_j,j in enumerate(i):

                # Get the indices of the previous and next atoms in the ring
                if count_j == 0:
                    prev_atom = i[len(i)-1]
                    next_atom = i[count_j + 1]
                elif count_j == len(i)-1:
                    prev_atom = i[count_j - 1]
                    next_atom = i[0]
                else:
                    prev_atom = i[count_j - 1]
                    next_atom = i[count_j + 1]

                # second check is to avoid starting on an allene
                if bond_mat[j,prev_atom] > 1 and bond_mat[j,next_atom] == 1:
                    if count_j % 2 == 0:
                        loop_ind = i[count_j::2] + i[:count_j:2]
                    else:
                        loop_ind = i[count_j::2] + i[1:count_j:2] # for an odd starting index the first index needs to be skipped
                    break

            # If a valid starting point was found
            if loop_ind:
                    
                # Loop over the atoms in the (anti)aromatic ring
                move = []
                for j in loop_ind:

                    # Get the indices of the previous and next atoms in the ring
                    if i.index(j) == 0:
                        prev_atom = i[len(i)-1]
                        next_atom = i[1]
                    elif i.index(j) == len(i)-1:
                        prev_atom = i[i.index(j) - 1]
                        next_atom = i[0]
                    else:
                        prev_atom = i[i.index(j) - 1]
                        next_atom = i[i.index(j) + 1]

                    # bonds are created in the forward direction.
                    if bond_mat[j,prev_atom] > 1:
                        move += [(-1,j,prev_atom),(-1,prev_atom,j),(1,j,next_atom),(1,next_atom,j)]

                    # If there is no double-bond (between j and the next or previous) then the shuffle does not apply.
                    # Note: lone pair and electron deficient aromatic moves are handled via Moves 3 and 1 above, respectively. Pi shuffles are only handled here.
                    else:
                        move = []
                        break

                # If a shuffle was generated then yield the move
                if move:
                    yield move


                    
# Helper function for valid_moves. This returns True is the proposed move increases aromaticity of at least one ring. 
def delta_aromatic(bond_mat,rings,move):
    tmp = copy(bond_mat)
    for k in move: tmp[k[1],k[2]]+=k[0]
    for r in rings:
        if ( is_aromatic(tmp,r) - is_aromatic(bond_mat,r) > 0):
            return True
    return False
    
# This is a simple version of valid_moves that only returns valid bond-formation moves with some quality checks (e.g., octet violations and allenes in rings)
# This is used to generate the initial guesses for the bond_mat    
def valid_bonds(ind,bond_mat,elements,reactive,ring_atoms):

    e = return_e(bond_mat) # current number of electrons associated with each atom

    # Check if a bond can be formed between neighbors ( electron available AND ( octet can be expanded OR octet is incomplete ))
    if bond_mat[ind,ind] > 0 and ( find_lewis.expand_octet[elements[ind]] or e[ind] < find_lewis.n_electrons[elements[ind]] ):
        # Check that ring contraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
        if ind not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[ind]) if count != ind and _ > 1 ]) == 0:  
           # Check on connected atoms
           for i in return_connections(ind,bond_mat,inds=reactive):
               # Electron available AND ( octect can be expanded OR octet is incomplete )
               if bond_mat[i,i] > 0 and ( find_lewis.expand_octet[elements[i]] or e[i] < find_lewis.n_electrons[elements[ind]] ):
                   # Check that ring contraints don't disqualify bond-formation ( not a ring atom OR no existing double/triple bonds )
                   if i not in ring_atoms or sum([ _ for count,_ in enumerate(bond_mat[i]) if count != i and _ > 1 ]) == 0:                  
                       return [(1,ind,i),(1,i,ind),(-1,ind,ind),(-1,i,i)]                                       
                
# bmat_score is the objective function that is minimized by the "best" lewis structures. The explanation of terms is as follows:
# 1st term: every electron deficiency (less than octet) is strongly penalized. electron deficiencies on more electronegative atoms are penalized more strongly.
# 2nd term: expanded octets are penalized at 0.1 per violation by default     
# 3rd term: formal charges are penalized based on their sign and the electronegativity of the atom they occur on
# 4th term: (anti)aromaticity is incentivized (penalized) depending on the size of the ring.  
def bmat_score(bond_mat,elements,rings,cat_en,an_en,rad_env,e_tet,w_def=-1,w_exp=0.1,w_formal=0.1,w_aro=-24,w_rad=0.1,factor=0.0,verbose=False):
    if verbose:
        print("deficiency: {}".format(w_def*sum([ _*cat_en[count] for count,_ in enumerate(return_def(bond_mat,elements,e_tet)) ])))
        print("expanded: {}".format(w_exp*sum(return_expanded(bond_mat,elements,e_tet))))
        print("formals: {}".format(w_formal*sum([ _*cat_en[count] if _ > 0 else 0.9 * _ * an_en[count] for count,_ in enumerate(return_formals(bond_mat,elements)) ])))
        print("aromatic: {}".format(w_aro*sum([ is_aromatic(bond_mat,_)/len(_) for _ in rings ])))
        print("radicals: {}".format(w_rad*sum([ rad_env[_]*(bond_mat[_,_]%2) for _ in range(len(bond_mat)) ])))
        print("factor: {}".format(factor))
        # objective function (lower is better): sum ( electron_deficiency * electronegativity_of_atom ) + sum ( expanded_octets ) + sum ( formal charge * electronegativity_of_atom ) + sum ( aromaticity of rings ) + factor
        return w_def*sum([ _*cat_en[count] for count,_ in enumerate(return_def(bond_mat,elements,e_tet)) ]) + \
              w_exp*sum(return_expanded(bond_mat,elements,e_tet)) + \
              w_formal*sum([ _ * cat_en[count] if _ > 0 else 0.9 * _ * an_en[count] for count,_ in enumerate(return_formals(bond_mat,elements)) ]) + \
              w_aro*sum([ is_aromatic(bond_mat,_)/len(_) for _ in rings ]) + \
              w_rad*sum([ rad_env[_]*(bond_mat[_,_]%2) for _ in range(len(bond_mat)) ]) + factor
    else:

        # objective function (lower is better): sum ( electron_deficiency * electronegativity_of_atom ) + sum ( expanded_octets ) + sum ( formal charge * electronegativity_of_atom ) + sum ( aromaticity of rings ) + factor
        return w_def*sum([ _*cat_en[count] for count,_ in enumerate(return_def(bond_mat,elements,e_tet)) ]) + \
              w_exp*sum(return_expanded(bond_mat,elements,e_tet)) + \
              w_formal*sum([ _ * cat_en[count] if _ > 0 else 0.9 * _ * an_en[count] for count,_ in enumerate(return_formals(bond_mat,elements)) ]) + \
              w_aro*sum([ is_aromatic(bond_mat,_)/len(_) for _ in rings ]) + \
              w_rad*sum([ rad_env[_]*(bond_mat[_,_]%2) for _ in range(len(bond_mat)) ]) + factor

# Returns 1,0,-1 for aromatic, non-aromatic, and anti-aromatic respectively
def is_aromatic(bond_mat,ring):

    # Initialize counter for pi electrons
    total_pi = 0

    # Loop over the atoms in the ring
    for count_i,i in enumerate(ring):

        # Get the indices of the previous and next atoms in the ring
        if count_i == 0:
            prev_atom = ring[len(ring)-1]
            next_atom = ring[count_i + 1]
        elif count_i == len(ring)-1:
            prev_atom = ring[count_i - 1]
            next_atom = ring[0]
        else:
            prev_atom = ring[count_i - 1]
            next_atom = ring[count_i + 1]

        # Check that there are pi electrons ( pi electrons on atom OR ( higher-order bond with ring neighbors) OR empty pi orbital
        if bond_mat[i,i] > 0 or ( bond_mat[i,prev_atom] > 1 or bond_mat[i,next_atom] > 1 ) or sum(bond_mat[i]) < 4:

            # Double-bonds are only counted with the next atom to avoid double counting. 
            if bond_mat[i,prev_atom] >= 2:
                total_pi += 0
            elif bond_mat[i,next_atom] >= 2:
                total_pi += 2
            # Handles carbenes: if only two bonds and there are less than three electrons then the orbital cannot participate in the pi system
#            elif (sum(bond_mat[i])-bond_mat[i,i])==2 and bond_mat[i,i] <= 2:
#                total_pi += 0
            # Elif logic is used, because if one of the previous occurs then the unbound electrons cannot be in the plane of the pi system.
            elif bond_mat[i,i] == 1:
                total_pi += 1
            elif bond_mat[i,i] >= 2:
                total_pi += 2

        # If there are no pi electrons then it is not an aromatic system
        else:
            return 0

    # If there isn't an even number of pi electrons it isn't aromatic/antiaromatic
    if total_pi % 2 != 0:
        return 0
    # The number of pi electron pairs needs to be less than the size of the ring for it to be aromatic
    # If this is excluded then spurious aromaticity can be observed for species like N1NN1    
    elif total_pi/2 >= len(ring):
        return 0
    # If the number of pi electron pairs is even then it is antiaromatic ring.
    elif total_pi/2 % 2 == 0:
        return -1
    # Else, the number of pi electron pairs is odd and it is an aromatic ring.
    else:
        return 1
    
# returns the valence electrons possessed by each atom (half of each bond)
def return_e(bond_mat):
    return np.sum(2*bond_mat,axis=1)-np.diag(bond_mat)
    
# returns the electron deficiencies of each atom (based on octet goal)
def return_def(bond_mat,elements,e_tet):
    tmp = np.sum(2*bond_mat,axis=1)-np.diag(bond_mat)-e_tet
    return np.where(tmp<0,tmp,0)
        
# returns the number of electrons in excess of the octet for each atom (based on octet goal)
def return_expanded(bond_mat,elements,e_tet):
    tmp = np.sum(2*bond_mat,axis=1)-np.diag(bond_mat)-e_tet
    return np.where(tmp>0,tmp,0)

# returns the formal charges on each atom
def return_formals(bond_mat,elements):
    return  np.array([find_lewis.valence[_] for _ in elements ]) - np.sum(bond_mat,axis=1)

# inds: optional subset of relevant atoms
def return_connections(ind,bond_mat,inds=None,min_order=1):
    if inds:
        return [ _ for _ in inds if bond_mat[ind,_] >= min_order and _ != ind ]
    else:
        return [ _ for count,_ in enumerate(bond_mat[ind]) if _ >= min_order and count != ind ]        
    
# This should be placed within taffi_functions.py
# return_ring_atom does most of the work, this function just cleans up the outputs and collates rings of different sizes
def return_rings(adj_list,max_size=20,remove_fused=True):

    # Identify rings
    rings=[]
    ring_size_list=range(max_size+1)[3:] # starts at 3
    for i in range(len(adj_list)):
        rings += [ _ for _ in return_ring_atoms(adj_list,i,ring_size=max_size,convert=False) if _ not in rings ]

    # Remove fused rings based on if another ring's atoms wholly intersect a given ring
    if remove_fused:
        del_ind = []
        for count_i,i in enumerate(rings):
            if count_i in del_ind:
                continue
            else:
                del_ind += [ count for count,_ in enumerate(rings) if count != count_i and count not in del_ind and  i.intersection(_) == i ]         
        del_ind = set(del_ind)        

        # ring_path is used to convert the ring sets into actual ordered sets of indices that create the ring
        rings = [ _ for count,_ in enumerate(rings) if count not in del_ind ]

    # ring_path is used to convert the ring sets into actual ordered sets of indices that create the ring.
    # rings are sorted by size
    rings = sorted([ ring_path(adj_list,_,path=None) for _ in rings ],key=len)

    # Return list of rings or empty list 
    if rings:
        return rings
    else:
        return []

# call main if this .py file is being called from the command line.
#if __name__ == "__main__":
#    main(sys.argv[1:])


# This function is used to deal with fragments. frag_find_lewis will just form double/triple bonds based on given fc/keep_lone information rather than generate fc/keep_lone.
# 
# Inputs:  elements:  a list of element labels indexed to the adj_mat 
#          adj_mat:   np.array of atomic connections
#          bonding_pref: optional list of (index, bond_number) tuples that sets the target bond number of the indexed atoms
#          q_tot:     total charge on the molecule
#          fixed_bonds: optional list of (index_1,index_2,bond_number) tuples that creates fixed bonds between the index_1
#                       and index_2 atoms. No further bonds will be added or subtracted between these atoms.
#
# Optional inputs for ion and radical cases:
#          fc_0:      a list of formal charges on each atom
#          keep_lone: a list of atom index for which contains a radical 
#
# Returns: lone_electrons:
#          bonding_electrons:
#          core_electrons:
#          bond_mat:  an NxN matrix holding the bond orders between all atoms in the adj_mat
#          bonding_pref (optinal): optional list of (index, bond_number) tuples that sets the target bond number of the indexed atoms  
#
def frag_find_lewis(elements,adj_mat_0,bonding_pref=[],fixed_bonds=[],q_tot=0,fc_0=None,keep_lone=[],return_pref=False,return_FC=False,octet_opt=True,check_lewis_flag=False):

    from itertools import combinations    

    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(frag_find_lewis, "sat_dict"):

        frag_find_lewis.lone_e = {'h':0, 'he':2,\
                             'li':0, 'be':2,                                                                                                                'b':0,     'c':0,     'n':2,     'o':4,     'f':6,    'ne':8,\
                             'na':0, 'mg':2,                                                                                                               'al':0,    'si':0,     'p':2,     's':4,    'cl':6,    'ar':8,\
                             'k':0, 'ca':2, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':10,    'ge':0,    'as':3,    'se':4,    'br':6,    'kr':None,\
                             'rb':0, 'sr':2,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':None, 'sn':None, 'sb':None, 'te':None,  'i':6,    'xe':None,\
                             'cs':0, 'ba':2, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':None, 'pb':None, 'bi':None, 'po':None, 'at':None, 'rn':None }

        # Initialize periodic table
        frag_find_lewis.periodic = { "h": 1,  "he": 2,\
                                 "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                                 "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                  "k":19, "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                                 "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                                 "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}
        
        # Electronegativity ordering (for determining lewis structure)
        frag_find_lewis.en = { "h" :2.3,  "he":4.16,\
                          "li":0.91, "be":1.58,                                                                                                               "b" :2.05, "c" :2.54, "n" :3.07, "o" :3.61, "f" :4.19, "ne":4.79,\
                          "na":0.87, "mg":1.29,                                                                                                               "al":1.61, "si":1.91, "p" :2.25, "s" :2.59, "cl":2.87, "ar":3.24,\
                          "k" :0.73, "ca":1.03, "sc":1.19, "ti":1.38, "v": 1.53, "cr":1.65, "mn":1.75, "fe":1.80, "co":1.84, "ni":1.88, "cu":1.85, "zn":1.59, "ga":1.76, "ge":1.99, "as":2.21, "se":2.42, "br":2.69, "kr":2.97,\
                          "rb":0.71, "sr":0.96, "y" :1.12, "zr":1.32, "nb":1.41, "mo":1.47, "tc":1.51, "ru":1.54, "rh":1.56, "pd":1.58, "ag":1.87, "cd":1.52, "in":1.66, "sn":1.82, "sb":1.98, "te":2.16, "i" :2.36, "xe":2.58,\
                          "cs":0.66, "ba":0.88, "la":1.09, "hf":1.16, "ta":1.34, "w" :1.47, "re":1.60, "os":1.65, "ir":1.68, "pt":1.72, "au":1.92, "hg":1.76, "tl":1.79, "pb":1.85, "bi":2.01, "po":2.19, "at":2.39, "rn":2.60} 

        # Polarizability ordering (for determining lewis structure)
        frag_find_lewis.pol ={ "h" :4.5,  "he":1.38,\
                          "li":164.0, "be":377,                                                                                                               "b" :20.5, "c" :11.3, "n" :7.4, "o" :5.3,  "f" :3.74, "ne":2.66,\
                          "na":163.0, "mg":71.2,                                                                                                              "al":57.8, "si":37.3, "p" :25.0,"s" :19.4, "cl":14.6, "ar":11.1,\
                          "k" :290.0, "ca":161.0, "sc":97.0, "ti":100.0, "v": 87.0, "cr":83.0, "mn":68.0, "fe":62.0, "co":55, "ni":49, "cu":47.0, "zn":38.7,  "ga":50.0, "ge":40.0, "as":30.0,"se":29.0, "br":21.0, "kr":16.8,\
                          "rb":320.0, "sr":197.0, "y" :162,  "zr":112.0, "nb":98.0, "mo":87.0, "tc":79.0, "ru":72.0, "rh":66, "pd":26.1, "ag":55, "cd":46.0,  "in":65.0, "sn":53.0, "sb":43.0,"te":28.0, "i" :32.9, "xe":27.3,}

        # Bond energy dictionary {}-{}-{} refers to atom1, atom2 additional bonds number (1 refers to double bonds)
        # From https://www2.chemistry.msu.edu/faculty/reusch/OrgPage/bndenrgy.htm 
        # Remove "6-16-1":73
        frag_find_lewis.be = { "6-6-1": 63, "6-6-2":117, "6-7-1":74, "6-7-2":140, "6-8-1":92.5, "6-8-2":172.5, "7-7-1":70.6, "7-7-2":187.6, "7-8-1":88, "8-8-1":84, "8-15-1":20, "8-16-1":6, "15-15-1":84,"15-15-2": 117, "15-16-1":70}

        # Initialize periodic table
        frag_find_lewis.atomic_to_element = { frag_find_lewis.periodic[i]:i for i in frag_find_lewis.periodic.keys() }

    # Consistency check on fc_0 argument, if supplied
    if fc_0 is not None:

        if len(fc_0) != len(elements):
            print("ERROR in frag_find_lewis: the fc_0 and elements lists must have the same dimensions.")
            quit()

        if int(sum(fc_0)) != int(q_tot):
            print("ERROR in frag_find_lewis: the sum of formal charges does not equal q_tot.")
            quit()

    # Initalize elementa and atomic_number lists for use by the function
    atomic_number = [ frag_find_lewis.periodic[i.lower()] for i in elements ]
    adj_mat = deepcopy(adj_mat_0)
    
    # calculate graph distance
    gs = graph_seps(adj_mat_0)
    
    # determine center atoms (atom with the smallest max_dis)
    distance = []
    for i in range(len(elements)):
        distance += [max(gs[i])]
    center = [i for i,dis in enumerate(distance) if dis == min(distance)]

    # Initially assign all valence electrons as lone electrons
    lone_electrons    = np.zeros(len(elements),dtype="int")    
    bonding_electrons = np.zeros(len(elements),dtype="int")    
    core_electrons    = np.zeros(len(elements),dtype="int")
    valence           = np.zeros(len(elements),dtype="int")
    bonding_target    = np.zeros(len(elements),dtype="int")
    valence_list      = np.zeros(len(elements),dtype="int")    
    
    for count_i,i in enumerate(elements):

        # Grab the total number of (expected) electrons from the atomic number
        N_tot = atomic_number[count_i]   

        # Determine the number of core/valence electrons based on row in the periodic table
        if N_tot > 54:
            print("ERROR in frag_find_lewis: the algorithm isn't compatible with atomic numbers greater than 54 owing to a lack of rules for treating lanthanides. Exiting...")
            quit()

        elif N_tot > 36:
            N_tot -= 36
            core_electrons[count_i] = 36
            valence[count_i]        = 18

        elif N_tot > 18:
            N_tot -= 18
            core_electrons[count_i] = 18
            valence[count_i]        = 18

        elif N_tot > 10:
            N_tot -= 10
            core_electrons[count_i] = 10
            valence[count_i]        = 8

        elif N_tot > 2:
            N_tot -= 2
            core_electrons[count_i] = 2
            valence[count_i]        = 8

        lone_electrons[count_i] = N_tot
        valence_list[count_i] = N_tot

        # Assign target number of bonds for this atom
        if count_i in [ j[0] for j in bonding_pref ]:
            bonding_target[count_i] = next( j[1] for j in bonding_pref if j[0] == count_i )
        else:
            bonding_target[count_i] = N_tot - frag_find_lewis.lone_e[elements[count_i].lower()]       

    # Loop over the adjmat and assign initial bonded electrons assuming single bonds (and adjust lone electrons accordingly)
    for count_i,i in enumerate(adj_mat_0):
        bonding_electrons[count_i] += sum(i)
        lone_electrons[count_i] -= sum(i)

    # Apply keep_lone: add one electron to such index    
    for count_i in keep_lone:
        lone_electrons[count_i] += 1
        bonding_target[count_i] -= 1

    # Eliminate all radicals by forming higher order bonds
    change_list = range(len(lone_electrons))
    bonds_made = []    
    loop_list   = [ (atomic_number[i],i) for i in range(len(lone_electrons)) ]
    loop_list   = [ i[1] for i in sorted(loop_list) ]

    # Loop over bonding_pref, find whether exist two of them can form bonds
    if len(bonding_pref) > 1:
        bonding_pref_ind = [i[0] for i in bonding_pref]
        comb = combinations(bonding_pref_ind, 2)
        pref_pair = [sorted(pair) for pair in comb if adj_mat_0[pair[0]][pair[1]] == 1]
    else:
        pref_pair = []

    # Check for special chemical groups
    for i in range(len(elements)):

        # Handle nitro groups
        if is_nitro(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ]
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],1)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            lone_electrons[O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[1],i] += 1

        # Handle sulfoxide groups
        if is_frag_sulfoxide(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the thioketone atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind[0])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[O_ind[0],i] += 1

        # Handle sulfonyl groups
        if is_frag_sulfonyl(i,adj_mat_0,elements) is True:
            
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the sulfoxide atoms from the bonding_pref list
            bonding_pref += [(i,6)]
            bonding_pref += [(O_ind[0],2)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 2
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            bonds_made += [(i,O_ind[0])]
            bonds_made += [(i,O_ind[1])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[0],i] += 1
            adj_mat[O_ind[1],i] += 1            
        
        # Handle phosphate groups 
        if is_phosphate(i,adj_mat_0,elements) is True:
            O_ind      = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] in ["o","O"] ] # Index of single bonded O-P oxygens
            O_ind_term = [ j for j in O_ind if sum(adj_mat_0[j]) == 1 ] # Index of double bonded O-P oxygens
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the phosphate atoms from the bonding_pref list
            bonding_pref += [(i,5)]
            bonding_pref += [(O_ind_term[0],2)]  # during testing it ended up being important to only add a bonding_pref tuple for one of the terminal oxygens
            bonding_electrons[O_ind_term[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind_term[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind_term[0])]
            adj_mat[i,O_ind_term[0]] += 1
            adj_mat[O_ind_term[0],i] += 1

        # Handle isocyano groups
        if is_isocyano(i,adj_mat,elements) is True:
            C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in  ["c","C"] and sum(adj_mat[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in C_ind ] # remove bonds involving the cyano atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(C_ind[0],3)]
            bonding_electrons[C_ind[0]] += 2
            bonding_electrons[i] += 2
            lone_electrons[C_ind[0]] -= 2
            lone_electrons[i] -= 2
            bonds_made += [(i,C_ind[0])]
            bonds_made += [(i,C_ind[0])]
            adj_mat[i,C_ind[0]] += 2
            adj_mat[C_ind[0],i] += 2
    
        # handle ethenone group
        if is_frag_ethenone(i,adj_mat,elements) is True:
            CN_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C","n","N"]][0]
            bonding_electrons[CN_ind] += 1
            bonding_electrons[i] += 1
            lone_electrons[CN_ind] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,CN_ind)]
            adj_mat[i,CN_ind] += 1
            adj_mat[CN_ind,i] += 1
                        
    # Apply fixed_bonds argument
    off_limits=[]
    for i in fixed_bonds:

        # Initalize intermediate variables
        a = i[0]
        b = i[1]
        N = i[2]
        N_current = len([ j for j in bonds_made if (a,b) == j or (b,a) == j ]) + 1
        # Check that a bond exists between these atoms in the adjacency matrix
        if adj_mat_0[a,b] != 1:
            print("ERROR in frag_find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but the adjacency matrix doesn't reflect a bond. Exiting...")
            quit()

        # Check that less than or an equal number of bonds exist between these atoms than is requested
        if N_current > N:
            print("ERROR in frag_find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but {} bonds already exist between these atoms. There may be a conflict".format(N_current))
            print("                      between the special groups handling and the requested lewis_structure.")
            quit()

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[a] < (N - N_current):
            print("Warning in frag_find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(elements[a],lone_electrons[a]))

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[b] < (N - N_current):
            print("Warning in frag_find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(elements[b],lone_electrons[b]))
        

        # Make the bonds between the atoms
        for j in range(N-N_current):
            bonding_electrons[a] += 1
            bonding_electrons[b] += 1
            lone_electrons[a]    -= 1
            lone_electrons[b]    -= 1
            bonds_made += [ (a,b) ]

        # Append bond to off_limits group so that further bond additions/breaks do not occur.
        off_limits += [(a,b),(b,a)]

    # Turn the off_limits list into a set for rapid lookup
    off_limits = set(off_limits)

    # Adjust formal charges (if supplied)
    if fc_0 is not None:
        for count_i,i in enumerate(fc_0):
            if i > 0:
                #if lone_electrons[count_i] < i:
                #    print "ERROR in find_lewis: atom ({}, index {}) doesn't have enough lone electrons ({}) to be removed to satisfy the specified formal charge ({}).".format(elements[count_i],count_i,lone_electrons[count_i],i)
                #    quit()
                lone_electrons[count_i] = lone_electrons[count_i] - i
            if i < 0:
                lone_electrons[count_i] = lone_electrons[count_i] + int(abs(i))
        q_tot=0
        
    # Initialize objects for use in the algorithm
    lewis_total = 1000
    lewis_lone_electrons = []
    lewis_bonding_electrons = []
    lewis_core_electrons = []
    lewis_valence = []
    lewis_bonding_target = []
    lewis_bonds_made = []
    lewis_adj_mat = []
    lewis_identical_mat = []
    
    # Determine the atoms with lone pairs that are unsatisfied as candidates for electron removal/addition to satisfy the total charge condition  
    happy = [ i[0] for i in bonding_pref if i[1] <= bonding_electrons[i[0]]]
    bonding_pref_ind = [i[0] for i in bonding_pref]
    
    # Determine is electrons need to be removed or added
    if q_tot > 0:
        adjust = -1
        octet_violate_e = []
        for count_j,j in enumerate(elements):
            if j.lower() in ["c","n","o","f","si","p","s","cl"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] > 8:
                    octet_violate_e += [count_j]
            elif j.lower() in ["br","i"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] > 18:
                    octet_violate_e += [count_j]
        
        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy and count_i not in octet_violate_e]

    elif q_tot < 0:
        adjust = 1
        octet_violate_e = []
        for count_j,j in enumerate(elements):
            if j.lower() in ["c","n","o","f","si","p","s","cl"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] < 8:
                    octet_violate_e += [count_j]
                    
            elif j.lower() in ["br","i"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] < 18:
                    octet_violate_e += [count_j]

        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy and count_i not in octet_violate_e]
        
    else:
        adjust = 1
        octet_violate_e = []
        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy ]
    
    # The outer loop checks each bonding structure produced by the inner loop for consistency with
    # the user specified "pref_bonding" and pref_argument with bonding electrons are
    for dummy_counter in range(lewis_total):
        lewis_loop_list = loop_list
        random.shuffle(lewis_loop_list)
        outer_counter     = 0
        inner_max_cycles  = 1000
        outer_max_cycles  = 1000
        bond_sat = False
        
        lewis_lone_electrons.append(deepcopy(lone_electrons))
        lewis_bonding_electrons.append(deepcopy(bonding_electrons))
        lewis_core_electrons.append(deepcopy(core_electrons))
        lewis_valence.append(deepcopy(valence))
        lewis_bonding_target.append(deepcopy(bonding_target))
        lewis_bonds_made.append(deepcopy(bonds_made))
        lewis_adj_mat.append(deepcopy(adj_mat))
        lewis_counter = len(lewis_lone_electrons) - 1
        
        # Adjust the number of electrons by removing or adding to the available lone pairs
        # The algorithm simply adds/removes from the first N lone pairs that are discovered
        random.shuffle(octet_violate_e)
        random.shuffle(normal_adjust)
        adjust_ind=octet_violate_e+normal_adjust
                
        if len(adjust_ind) >= abs(q_tot): 
            for i in range(abs(q_tot)):
                lewis_lone_electrons[-1][adjust_ind[i]] += adjust
                lewis_bonding_target[-1][adjust_ind[i]] += adjust 
        else:
            for i in range(abs(q_tot)):
                lewis_lone_electrons[-1][0] += adjust
                lewis_bonding_target[-1][0] += adjust
                
        # Search for an optimal lewis structure
        while bond_sat is False:
        
            # Initialize necessary objects
            change_list   = range(len(lewis_lone_electrons[lewis_counter]))
            inner_counter = 0
            bond_sat = True                
            # Inner loop forms bonds to remove radicals or underbonded atoms until no further
            # changes in the bonding pattern are observed.
            while len(change_list) > 0:
                change_list = []
                for i in lewis_loop_list:

                    # List of atoms that already have a satisfactory binding configuration.
                    happy = [ j[0] for j in bonding_pref if j[1] <= lewis_bonding_electrons[lewis_counter][j[0]]]            
                    
                    # If the current atom already has its target configuration then no further action is taken
                    if i in happy: continue

                    # If there are no lone electrons or too more bond formed then skip
                    if lewis_lone_electrons[lewis_counter][i] == 0: continue
                    
                    # Take action if this atom has a radical or an unsatifisied bonding condition
                    if lewis_lone_electrons[lewis_counter][i] % 2 != 0 or lewis_bonding_electrons[lewis_counter][i] != lewis_bonding_target[lewis_counter][i]:
                        # Try to form a bond with a neighboring radical (valence +1/-1 check ensures that no improper 5-bonded atoms are formed)
                        lewis_bonded_lonepairs= [ (-frag_find_lewis.en[elements[count_j].lower()],count_j) for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and lewis_lone_electrons[lewis_counter][count_j] > 0 \
                                                  and 2*(lewis_bonding_electrons[lewis_counter][count_j]+1)+(lewis_lone_electrons[lewis_counter][count_j]-1) <= lewis_valence[lewis_counter][count_j] and\
                                                  lewis_lone_electrons[lewis_counter][count_j]-1 >= 0 and count_j not in happy ]

                        # Sort by atomic number (cheap way of sorting carbon before other atoms, should probably switch over to electronegativities) 
                        lewis_bonded_lonepairs = [ j[1] for j in sorted(lewis_bonded_lonepairs) ]
                            
                        # Try to form a bond with a neighboring atom with spare lone electrons (valence check ensures that no improper 5-bonded atoms are formed)
                        if len(lewis_bonded_lonepairs) > 0:
                            lewis_bonding_electrons[lewis_counter][i] += 1
                            lewis_bonding_electrons[lewis_counter][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][i][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][lewis_bonded_lonepairs[0]][i] += 1
                            lewis_lone_electrons[lewis_counter][i] -= 1
                            lewis_lone_electrons[lewis_counter][lewis_bonded_lonepairs[0]] -= 1
                            change_list += [i,lewis_bonded_lonepairs[0]]
                            lewis_bonds_made[lewis_counter] += [(i,lewis_bonded_lonepairs[0])]
                                
                # Increment the counter and break if the maximum number of attempts have been made
                inner_counter += 1
                if inner_counter >= inner_max_cycles:
                    print("WARNING: maximum attempts to establish a reasonable lewis-structure exceeded ({}).".format(inner_max_cycles))
            
            # Check if the user specified preferred bond order has been achieved.
            if bonding_pref is not None:
                unhappy = [ i[0] for i in bonding_pref if i[1] != lewis_bonding_electrons[lewis_counter][i[0]]]            
                if len(unhappy) > 0:

                    # Break the first bond involving one of the atoms bonded to the under/over coordinated atoms
                    ind = set([unhappy[0]] + [ count_i for count_i,i in enumerate(adj_mat_0[unhappy[0]]) if i == 1 and (count_i,unhappy[0]) not in off_limits ])
                    
                    potential_bond = [i for i in lewis_bonds_made[lewis_counter] if ( (i[0] in ind or i[1] in ind ) and (i[0] not in bonding_pref_ind or i[1] not in bonding_pref_ind) ) ]  
                    if len(potential_bond) == 0 :                                                                                                                                                          
                        potential_bond = [i for i in lewis_bonds_made[lewis_counter] if i[0] in ind or i[1] in ind  ]                                                                                          

                    # Check if a rearrangment is possible, break if none are available
                    try:
                        break_bond = next( i for i in potential_bond ) 
                    except:
                        
                        print("WARNING: no further bond rearrangments are possible and bonding_pref is still not satisfied.")
                        print(Qiyuan)
                        break
                    
                    # Perform bond rearrangment
                    lewis_bonding_electrons[lewis_counter][break_bond[0]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[0]] += 1
                    lewis_adj_mat[lewis_counter][break_bond[0]][break_bond[1]] -= 1
                    lewis_adj_mat[lewis_counter][break_bond[1]][break_bond[0]] -= 1
                    lewis_bonding_electrons[lewis_counter][break_bond[1]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[1]] += 1

                    # Remove the bond from the list and reorder lewis_loop_list so that the indices involved in the bond are put last                
                    lewis_bonds_made[lewis_counter].remove(break_bond)
                    lewis_loop_list.remove(break_bond[0])
                    lewis_loop_list.remove(break_bond[1])
                    lewis_loop_list += [break_bond[0],break_bond[1]]
        
                    # Update the bond_sat flag
                    bond_sat = False
                    
                # Increment the counter and break if the maximum number of attempts have been made
                outer_counter += 1
                    
                # Periodically reorder the list to avoid some cyclical walks
                if outer_counter % 100 == 0:
                    lewis_loop_list = reorder_list(lewis_loop_list,atomic_number)

                # Print diagnostic upon failure
                if outer_counter >= outer_max_cycles:
                    print("WARNING: maximum attempts to establish a lewis-structure consistent")
                    print("         with the user supplied bonding preference has been exceeded ({}).".format(outer_max_cycles))
                    print(Qiyuan)
                    break
                
        # Re-apply keep_lone: remove one electron from such index  
        for count_i in keep_lone:
            lewis_lone_electrons[lewis_counter][count_i] -= 1

        # Delete last entry in the lewis np.arrays if the electronic structure is not unique
        identical_mat=np.vstack([lewis_adj_mat[-1], np.array([ valence_list[k] - lewis_bonding_electrons[-1][k] - lewis_lone_electrons[-1][k] for k in range(len(elements)) ]) ])
        lewis_identical_mat.append(identical_mat)

        if array_unique(lewis_identical_mat[-1],lewis_identical_mat[:-1])[0] is False :
            lewis_lone_electrons    = lewis_lone_electrons[:-1]
            lewis_bonding_electrons = lewis_bonding_electrons[:-1]
            lewis_core_electrons    = lewis_core_electrons[:-1]
            lewis_valence           = lewis_valence[:-1]
            lewis_bonding_target    = lewis_bonding_target[:-1]
            lewis_bonds_made        = lewis_bonds_made[:-1]
            lewis_adj_mat           = lewis_adj_mat[:-1]
            lewis_identical_mat     = lewis_identical_mat[:-1]
            
    # Find the total number of lone electrons in each structure
    lone_electrons_sums = []
    for i in range(len(lewis_lone_electrons)):
        lone_electrons_sums.append(sum(lewis_lone_electrons[i]))

    # Find octet violations in each structure
    octet_violations = []
    for i in range(len(lewis_lone_electrons)):
        ov = 0
        if octet_opt is True:
            for count_j,j in enumerate(elements):
                if j.lower() in ["c","n","o","f","si","p","s","cl"] and count_j not in bonding_pref_ind:
                    if lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j] < 8:
                        ov += (8 - lewis_bonding_electrons[i][count_j]*2 - lewis_lone_electrons[i][count_j])
                    else:
                        ov += 2 * (lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j]-8)
                if j.lower() in ["br",'i'] and count_j not in bonding_pref_ind:
                    if lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j] < 18:
                        ov += (18 - lewis_bonding_electrons[i][count_j]*2 - lewis_lone_electrons[i][count_j])
                    else:
                        ov += 2 * (lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j]-18)
        octet_violations.append(ov)
    
    # Find the total formal charge for each structure
    formal_charges_sums = []
    for i in range(len(lewis_lone_electrons)):
        fc = 0
        for j in range(len(elements)):
            fc += valence_list[j] - lewis_bonding_electrons[i][j] - lewis_lone_electrons[i][j]
        formal_charges_sums.append(fc)
                                   
    # Find formal charge eletronegativity contribution
    lewis_fc_en   = [] # Electronegativity for stabling charge/radical
    lewis_fc_pol  = [] # Polarizability for stabling charge/radical
    lewis_fc_hc   = [] # Hyper-conjugation contribution 
    lewis_fc_bond = [] # First form bond with radical/charge
    formal_charge = deepcopy(fc_0)
    radical_atom  = deepcopy(keep_lone)

    # Consider the facts that will stabilize ions/radicals
    for i in range(len(lewis_lone_electrons)):
        fc_ind = [(count_j,j) for count_j,j in enumerate(formal_charge) if j != 0]
        for R_ind in radical_atom:  # assign +0.5 for radical
            fc_ind += [(R_ind,0.5)]
        
        # initialize en,pol and hc
        fc_bond,fc_en,fc_pol,fc_hc = 0,0,0,0

        # Loop over formal charges and radicals
        for count_fc in fc_ind:

            ind = count_fc[0]
            charge = count_fc[1]

            # find bonds formed
            bonds_made = lewis_bonds_made[i]
            if True in [count_fc[0] in bm for bm in bonds_made]:
                bond_atom = [ba for ba in bonds_made[[count_fc[0] in bm for bm in bonds_made].index(True)] if ba != count_fc[0]][0]
                if bond_atom in center:
                    fc_bond -= 1
                else:
                    fc_bond -= 0.5
                    
            # Count the self contribution: (-) on the most electronegative atom and (+) on the least electronegative atom
            fc_en += 10 * charge * frag_find_lewis.en[elements[ind].lower()]

            # Find the nearest and next-nearest atoms for each formal_charge/radical contained atom
            nearest_atoms = [count_k for count_k,k in enumerate(lewis_adj_mat[i][ind]) if k >= 1] 
            NN_atoms = list(set([ count_j for count_j,j in enumerate(gs[ind]) if j == 2 ]))
            
            # only count en > en(C)
            fc_en += charge*(sum([frag_find_lewis.en[elements[count_k].lower()] for count_k in nearest_atoms if frag_find_lewis.en[elements[count_k].lower()] > 2.54] )+\
                             sum([frag_find_lewis.en[elements[count_k].lower()] for count_k in NN_atoms if frag_find_lewis.en[elements[count_k].lower()] > 2.54] ) * 0.1 )

            if charge < 0: # Polarizability only affects negative charge ?
                fc_pol += charge*sum([frag_find_lewis.pol[elements[count_k].lower()] for count_k in nearest_atoms ])

            # find hyper-conjugation strcuture
            nearby_carbon = [nind for nind in nearest_atoms if elements[nind].lower()=='c']
            for carbon_ind in nearby_carbon:
                carbon_nearby=[nind for nind in NN_atoms if lewis_adj_mat[i][carbon_ind][nind] >= 1 and elements[nind].lower() in ['c','h']]
                radical_on_carbon = lewis_lone_electrons[i][carbon_ind]
                if (len(carbon_nearby)+radical_on_carbon) == 3: fc_hc -= charge*(len([nind for nind in carbon_nearby if elements[nind].lower() == 'c']) * 2 +\
                                                                                 len([nind for nind in carbon_nearby if elements[nind].lower() == 'h']) + radical_on_carbon )
            
        lewis_fc_en.append(fc_en)        
        lewis_fc_pol.append(fc_pol)        
        lewis_fc_hc.append(fc_hc)        
        lewis_fc_bond.append(fc_bond)        

    # normalize the effect
    lewis_fc_en = [lfc/max(1,max(abs(np.array(lewis_fc_en)))) for lfc in lewis_fc_en]
    lewis_fc_pol= [lfp/max(1,max(abs(np.array(lewis_fc_pol)))) for lfp in lewis_fc_pol]

    # The bond formation will try to keep resonance structures
    # First find identical atoms; then determine whether the distance between them is 2, if so, find the connecting atom to this pair to form two "resonance preferance pairs". After
    # finding all such pairs, count it in lewis critria.
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}
    masses    = [ mass_dict[i] for i in elements ]
    hash_list = [  atom_hash(ind,adj_mat_0,masses) for ind in range(len(elements)) ]

    # Find identical atoms
    same_atoms= [(i,[ count_j for count_j,j in enumerate(hash_list) if j == i ]  ) for i in set(hash_list) if hash_list.count(i) > 1]
    gs = graph_seps(adj_mat_0) 
    res_atoms = []
    
    # Keep identical atoms whose distance = 2
    for i in same_atoms:
        if len(i[1]) == 2 and gs[i[1][0]][i[1][1]] == 2: res_atoms += [ [tuple(i[1]),2]]
        if len(i[1]) > 2:
            comb = combinations(i[1], 2)
            res_atoms += [ [j,len(i[1])] for j in comb if gs[j[0]][j[1]] == 2 ]
    
    # Find connecting atom to form resonance pairs
    res_pair = []
    res_pair_score = []
    for pair_list in res_atoms:
        pair = pair_list[0]
        center_atom = [ ind for ind,i in enumerate(adj_mat_0[pair[0]]) if i==1 and adj_mat_0[pair[1]][ind] ==1 ][0]
        res_pair += [ tuple(sorted((pair[0],center_atom))),tuple(sorted((pair[1],center_atom)))]
        res_pair_score += [pair_list[1],pair_list[1]]
    
    # loop over lewis_bonds_made to determine the lewis criteria
    lewis_res_bonds = []    
    lewis_bonds_energy = []
    for bonds_made in lewis_bonds_made:
        res_bonds = 0
        bonds_energy = 0
        for lb,bond_made in enumerate(bonds_made): bonds_made[lb]=tuple(sorted(bond_made))
        for bond_made in bonds_made:
            # in the bonds in res_pair, enlarge by the number of symmetry atoms
            if bond_made in res_pair:
                res_bonds += res_pair_score[res_pair.index(bond_made)]
            # if the bonds in pref_bonds, enlarge the effect by 2; for cation, prefer to form bonds while for anion/radical prefer to keep single bond
            factor = 0.1 # make bonds_energy comparable with originally used bonds_en
            if sorted([bond_made[0],bond_made[1]]) in pref_pair : factor *= 2 
            if fc_0[bond_made[0]] > 0 or fc_0[bond_made[1]] > 0 : factor *= 2 
            if fc_0[bond_made[0]] < 0 or fc_0[bond_made[1]] < 0 : factor *= 0.5
            if bond_made[0] in keep_lone or bond_made[1] in keep_lone : factor *= 0.5
            bond_type = "{}-{}-{}".format(min(atomic_number[bond_made[0]],atomic_number[bond_made[1]]),max(atomic_number[bond_made[0]],atomic_number[bond_made[1]]),bonds_made.count(bond_made))
            if bond_type in frag_find_lewis.be.keys(): bonds_energy += factor * frag_find_lewis.be[bond_type]
            else: bonds_energy -= factor * (-10000.0)

        lewis_bonds_energy += [bonds_energy]
        lewis_res_bonds += [res_bonds]

    # normalize the effect
    lewis_bonds_energy = [-be/max(1,max(lewis_bonds_energy)) for be in lewis_bonds_energy]
    lewis_res_bonds    = [-re/max(1,max(lewis_res_bonds)) for re in lewis_res_bonds]
    
    # Add the total number of radicals to the total formal charge to determine the criteria.
    # The radical count is scaled by 0.01 and the lone pair count is scaled by 0.001. This results
    # in the structure with the lowest formal charge always being returned, and the radical count 
    # only being considered if structures with equivalent formal charges are found, and likewise with
    # the lone pair count. The structure(s) with the lowest score will be returned.
    lewis_criteria = []
    for i in range(len(lewis_lone_electrons)):
        lewis_criteria.append( 10.0 * octet_violations[i] + lewis_fc_en[i] + lewis_fc_pol[i] + 0.5 * lewis_fc_bond[i] + 0.1 * lewis_fc_hc[i] + 0.01 * lewis_bonds_energy[i] + 0.0001 * lewis_res_bonds[i] )
    #print(lewis_bonds_made)
    #print(lewis_criteria)
    #print(lewis_fc_en,lewis_fc_pol,lewis_fc_bond,lewis_bonds_energy)
    #exit()
    best_lewis = [i[0] for i in sorted(enumerate(lewis_criteria), key=lambda x:x[1])]  # sort from least to most and return a list containing the origial list's indices in the correct order
    best_lewis = [ i for i in best_lewis if lewis_criteria[i] == lewis_criteria[best_lewis[0]] ]

    # Apply keep_lone information, remove the electron to form lone electron
    for i in best_lewis:
        for j in keep_lone:
            lewis_lone_electrons[i][j] -= 1

    # return check_lewis function
    if check_lewis_flag is True:
        if return_pref is True:
            return lewis_lone_electrons[best_lewis[0]], lewis_bonding_electrons[best_lewis[0]], lewis_core_electrons[best_lewis[0]],bonding_pref
        else:
            return lewis_lone_electrons[best_lewis[0]], lewis_bonding_electrons[best_lewis[0]], lewis_core_electrons[best_lewis[0]]
    
    # Optional bonding pref return to handle cases with special groups
    if return_pref is True:
        if return_FC is False:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],bonding_pref
        else:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],\
                   [ [ valence_list[i] - lewis_bonding_electrons[_][i] - lewis_lone_electrons[_][i] for i in range(len(elements)) ] for _ in best_lewis ],bonding_pref

    else:
        if return_FC is False:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ]
        else:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],\
                   [ [ valence_list[i] - lewis_bonding_electrons[_][i] - lewis_lone_electrons[_][i] for i in range(len(elements)) ] for _ in best_lewis ]
    
# Description: Function to determine whether given atom index in the input sturcture locates on a ring or not
#
# Inputs      adj_mat:   NxN array holding the molecular graph
#             idx:       atom index
#             ring_size: number of atoms in a ring
#
# Returns     Bool value depending on if idx is a ring atom 
#
def ring_atom(adj_mat,idx,start=None,ring_size=10,counter=0,avoid_set=None,in_ring=None):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False,[]

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    if avoid_set is None:
        avoid_set = set([])
    if in_ring is None:
        in_ring=set([idx])

    # Trick: The fact that the smallest possible ring has three nodes can be used to simplify
    #        the algorithm by including the origin in avoid_set until after the second step
    if counter >= 2 and start in avoid_set:
        avoid_set.remove(start)    
    elif counter < 2 and start not in avoid_set:
        avoid_set.add(start)

    # Update the avoid_set with the current idx value
    avoid_set.add(idx)    
    
    # Loop over connections and recursively search for idx
    status = 0
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid_set ]
    
    if len(cons) == 0:
        return False,[]
    elif start in cons:
        return True,in_ring
    else:
        for i in cons:
            if ring_atom(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid_set=avoid_set,in_ring=in_ring)[0] == True:
                in_ring.add(i)
                return True,in_ring
        return False,[]


# Description: Canonicalizes the ordering of atoms in a geometry based on a hash function. Atoms that hash to equivalent values retain their relative order from the input geometry.
#
# Inputs:     elements:  a list of element labels indexed to the adj_mat 
#             adj_mat:   np.array of atomic connections
#
# Optional:   geo:       np.array of geometry
#             bond_mat:  np.array of bonding information
#
# Returns     Sorted inputs
#
def canon_geo(elements,adj_mat,geo=None,bond_mat=None,dup=[],change_group_seq=True):
    
    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    # Canonicalize by sorting the elements based on hashing
    masses = [ mass_dict[i] for i in elements ]
    hash_list = [ atom_hash(i,adj_mat,masses) for i in range(len(elements)) ]

    # determine the seperate compounds
    gs = graph_seps(adj_mat)
    groups = []
    loop_ind = []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
            loop_ind += new_group
            groups   += [new_group]

    # sort groups based on the maximum hash value
    if change_group_seq:
        _,group_seq = [list(k) for k in list(zip(*sorted([ (max([hash_list[j] for j in group]),lg) for lg,group in enumerate(groups) ], reverse=True)))]
        groups = [groups[i] for i in group_seq]

    # sort atoms in each group
    atoms = []
    for group in groups:
        _,seq  =  [ list(j) for j in list(zip(*sorted([ (hash_list[i],i) for i in group ],reverse=True)) )]
        atoms += seq
    
    # Update lists/arrays based on atoms
    adj_mat   = adj_mat[atoms]
    adj_mat   = adj_mat[:,atoms]
    elements  = [ elements[i] for i in atoms ]
    hash_list = [ hash_list[j] for j in atoms ]
              
    if geo is not None:
        geo   = geo[atoms]

    if bond_mat is not None:
        if len(bond_mat) == len(elements):
            bond_mat = bond_mat[atoms]
            bond_mat = bond_mat[:,atoms]
        else:
            for i in range(len(bond_mat)):
                bond_mat[i] = bond_mat[i][atoms] 
                bond_mat[i] = bond_mat[i][:,atoms]
    
    # Duplicate the respective lists
    if len(dup) > 0:
        N_dup = {}
        for count_i,i in enumerate(dup):
            N_dup[count_i] = []
            for j in atoms:
                N_dup[count_i] += [i[j]]
        N_dup = [ N_dup[i] for i in range(len(N_dup.keys())) ]

        if bond_mat is not None and geo is not None:
            return elements,adj_mat,hash_list,geo,bond_mat,N_dup
        elif bond_mat is None and geo is not None:
            return elements,adj_mat,hash_list,geo,N_dup
        elif bond_mat is not None and geo is None:
            return elements,adj_mat,hash_list,bond_mat,N_dup
        else:
            return elements,adj_mat,hash_list,N_dup
    else:
        if bond_mat is not None and geo is not None:
            return elements,adj_mat,hash_list,geo,bond_mat
        elif bond_mat is None and geo is not None:
            return elements,adj_mat,hash_list,geo
        elif bond_mat is not None and geo is None:
            return elements,adj_mat,hash_list,bond_mat
        else:
            return elements,adj_mat,hash_list


# Description: Hashing function for canonicalizing geometries on the basis of their adjacency matrices and elements
#
# Inputs      ind  : index of the atom being hashed
#             A    : adjacency matrix
#             M    : masses of the atoms in the molecule
#             gens : depth of the search used for the hash   
#
# Returns     hash value of given atom
#
def atom_hash(ind,A,M,alpha=100.0,beta=0.1,gens=10):    
    if gens <= 0:
        return rec_sum(ind,A,M,beta,gens=0)
    else:
        return alpha * sum(A[ind]) + rec_sum(ind,A,M,beta,gens)

# recursive function for summing up the masses at each generation of connections. 
def rec_sum(ind,A,M,beta,gens,avoid_list=[]):
    if gens != 0:
        tmp = M[ind]*beta
        new = [ count_j for count_j,j in enumerate(A[ind]) if j == 1 and count_j not in avoid_list ]
        if len(new) > 0:
            for i in new:
                tmp += rec_sum(i,A,M,beta*0.1,gens-1,avoid_list=avoid_list+[ind])
            return tmp
        else:
            return tmp
    else:
        return M[ind]*beta

# # Return true if idx is a ring atom
# # The algorithm spawns non-backtracking walks along the graph. If the walk encounters the starting node, that consistutes a cycle.        
def return_ring_atom(adj_mat,idx,start=None,ring_size=10,counter=0,avoid_set=None,in_ring=None):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")

    if counter == ring_size:
        return False,[]

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx

    if avoid_set is None:
        avoid_set = set([])

    if in_ring is None:
        in_ring=set([idx])

    # Trick: The fact that the smallest possible ring has three nodes can be used to simplify
    #        the algorithm by including the origin in avoid_set until after the second step
    if counter >= 2 and start in avoid_set:
        avoid_set.remove(start)    

    elif counter < 2 and start not in avoid_set:
        avoid_set.add(start)

    # Update the avoid_set with the current idx value
    avoid_set.add(idx)    
    
    # Loop over connections and recursively search for idx
    status = 0
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid_set ]
    #print cons,counter,start,avoid_set
    if len(cons) == 0:
        return False,[]

    elif start in cons:
        return True,in_ring

    else:
        for i in cons:
            if return_ring_atom(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid_set=avoid_set,in_ring=in_ring)[0] == True:
                in_ring.add(i)
                return True,in_ring
        return False,[]

# Return ring(s) that atom idx belongs to
# The algorithm spawns non-backtracking walks along the graph. If the walk encounters the starting node, that consistutes a cycle.        
def return_ring_atoms(adj_list,idx,start=None,ring_size=10,counter=0,avoid_set=None,convert=True):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")

    # Break if search has been exhausted
    if counter == ring_size:
        return []

    # Automatically assign start to the supplied idx value. For recursive calls this is updated each call
    if start is None:
        start = idx

    # Initially set to an empty set, during recursion this is occupied by already visited nodes
    if avoid_set is None:
        avoid_set = set([])

    # Trick: The fact that the smallest possible ring has three nodes can be used to simplify
    #        the algorithm by including the origin in avoid_set until after the second step
    if counter >= 2 and start in avoid_set:
        avoid_set.remove(start)    

    elif counter < 2 and start not in avoid_set:
        avoid_set.add(start)

    # Update the avoid_set with the current idx value
    avoid_set.add(idx)    

    # grab current connections while avoiding backtracking
    cons = adj_list[idx].difference(avoid_set)

    # You have run out of graph
    if len(cons) == 0:
        return []

    # You discovered the starting point
    elif start in cons:
        avoid_set.add(start)
        return [avoid_set]

    # The search continues
    else:
        rings = []
        for i in cons:
            rings = rings + [ i for i in return_ring_atoms(adj_list,i,start=start,ring_size=ring_size,counter=counter+1,avoid_set=copy(avoid_set),convert=convert) if i not in rings ]

    # Return of the original recursion is list of lists containing ordered atom indices for each cycle
    if counter==0:
        if convert:
            return [ ring_path(adj_list,_) for _ in rings ]
        else:
            return rings
            
    # Return of the other recursions is a list of index sets for each cycle (sets are faster for comparisons)
    else:
        return rings

# Convenience function for generating an ordered sequence of indices that enumerate a ring starting from the set of ring indices generated by return_ring_atoms()
def ring_path(adj_list,ring,path=None):

    # Initialize the loop starting from the minimum index, with the traversal direction set by min bonded index.
    if path is None:
        path = [min(ring),min(adj_list[min(ring)].intersection(ring))]

    # This for recursive construction is needed to handle branching possibilities. All branches are followed and only the one yielding the full cycle is returned
    for i in [ _ for _ in adj_list[path[-1]] if _ in ring and _ not in path ]:
        try:
            path = ring_path(adj_list,ring,path=path + [i])
            return path
        except:
            pass

    # Eventually the recursions will reach the end of a cycle (i.e., for i in []: for the above loop) and hit this.
    # If the path is shorter than the full cycle then it is invalid (i.e., the wrong branch was followed somewhere)        
    if len(path) == len(ring):
        return path
    else:
        raise Exception("wrong path, didn't recover ring") # This never gets printed, it is just used to trigger the except at a higher level of recursion. 
        
# Convenience function for converting between adjacency matrix and adjacency list (actually a list of sets for convenience)
def adjmat_to_adjlist(adj_mat):
    return [ set(np.where(_ == 1)[0]) for _ in adj_mat ]
        
# Return bool depending on if the atom is a nitro nitrogen atom
def is_nitro(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    if len(O_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfoxide sulfur atom
def is_sulfoxide(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] ] 
    if len(O_ind) == 1 and len(C_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_sulfonyl(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] ] 
    if len(O_ind) == 2 and len(C_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a phosphate phosphorus atom
def is_phosphate(i,adj_mat,elements):

    status = False
    if elements[i] not in ["P","p"]:
        return False
    O_ind      = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] ] 
    O_ind_term = [ j for j in O_ind if sum(adj_mat[j]) == 1 ]
    if len(O_ind) == 4 and sum(adj_mat[i]) == 4 and len(O_ind_term) > 0:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_cyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 2 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_isocyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 1 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

    # Return bool depending on if the atom is a sulfoxide sulfur atom
def is_frag_sulfoxide(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1] 
    connect = sum(adj_mat[i])

    if len(O_ind) >= 1 and int(connect) == 3:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_frag_sulfonyl(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1] 
    connect = sum(adj_mat[i])

    if len(O_ind) >= 2 and int(connect) == 4:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_frag_ethenone(i,adj_mat,elements):

    status = False
    if elements[i] not in ["C","c"]:
        return False

    OS_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O","s","S"] and sum(adj_mat[count_j]) == 1] 
    CN_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C","n","N"] ] 
    connect = sum(adj_mat[i])
    
    if len(OS_ind) == 1 and len(CN_ind) == 1 and int(connect) == 2:
        return True
    else:
        return False

# Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix 
def graph_seps(adj_mat_0):

    # Create a new name for the object holding A**(N), initialized with A**(1)
    adj_mat = deepcopy(adj_mat_0)
    
    # Initialize an array to hold the graphical separations with -1 for all unassigned elements and 0 for the diagonal.
    seps = np.ones([len(adj_mat),len(adj_mat)])*-1
    np.fill_diagonal(seps,0)

    # Perform searches out to len(adj_mat) bonds (maximum distance for a graph with len(adj_mat) nodes
    for i in np.arange(len(adj_mat)):        

        # All perform assignments to unassigned elements (seps==-1) 
        # and all perform an assignment if the value in the adj_mat is > 0        
        seps[np.where((seps==-1)&(adj_mat>0))] = i+1

        # Since we only care about the leading edge of the search and not the actual number of paths at higher orders, we can 
        # set the larger than 1 values to 1. This ensures numerical stability for larger adjacency matrices.
        adj_mat[np.where(adj_mat>1)] = 1
        
        # Break once all of the elements have been assigned
        if -1 not in seps:
            break

        # Take the inner product of the A**(i+1) with A**(1)
        adj_mat = np.dot(adj_mat,adj_mat_0)

    return seps

# Description: This function calls obminimize (open babel geometry optimizer function) to optimize the current geometry
#
# Inputs:      geo:      Nx3 array of atomic coordinates
#              adj_mat:  NxN array of connections
#              elements: N list of element labels
#              ff:       force-field specification passed to obminimize (uff, gaff)
#               q:       total charge on the molecule   
#
# Returns:     geo:      Nx3 array of optimized atomic coordinates
# 
def opt_geo(geo,adj_mat,elements,q=0,ff='mmff94',step=100,filename='tmp'):

    # Write a temporary molfile for obminimize to use
    tmp_filename = '.{}.mol'.format(filename)
    tmp_xyz_file = '.{}.xyz'.format(filename)
    count = 0
    while os.path.isfile(tmp_filename):
        count += 1
        if count == 10:
            print("ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_filename = ".{}".format(filename) + tmp_filename            

    counti = 0
    while os.path.isfile(tmp_xyz_file):
        counti += 1
        if counti == 10:
            print("ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_xyz_file = ".{}".format(filename) + tmp_xyz_file

    # Use the mol_write function imported from the write_functions.py 
    # to write the current geometry and topology to file
    elements_lower = [ _.lower() for _ in elements ]
    mol_write(tmp_filename,elements_lower,geo,adj_mat,q=q,append_opt=False)
    
    # Popen(stdout=PIPE).communicate() returns stdout,stderr as strings
    try:
        substring = 'obabel {} -O {} --sd --minimize --steps {} --ff {}'.format(tmp_filename,tmp_xyz_file,step,ff)
        output = subprocess.Popen(substring, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,bufsize=-1).communicate()[1].decode('utf8')
        element,geo = xyz_parse(tmp_xyz_file)

    except:
        substring = 'obabel {} -O {} --sd --minimize --steps {} --ff uff'.format(tmp_filename,tmp_xyz_file,step)
        output = subprocess.Popen(substring, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,bufsize=-1).communicate()[1].decode('utf8')
        element,geo = xyz_parse(tmp_xyz_file)

    # Remove the tmp file that was read by obminimize
    try:
        os.remove(tmp_filename)
        os.remove(tmp_xyz_file)
    except:
        pass

    return geo[:len(elements)]

# Description: Simple wrapper function for writing xyz file
#
# Inputs      name:     string holding the filename of the output
#             elements: list of element types (list of strings)
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
# Returns     None
#
def xyz_write(name,elements,geo,q_tot=0,append_opt=False,comment=''):

    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'
    if q_tot!=0:
        comment='q {}'.format(int(q_tot))
    with open(name,open_cond) as f:
        f.write('{}\n'.format(len(elements)))
        f.write('{}\n'.format(comment))
        for count_i,i in enumerate(elements):
            f.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i,geo[count_i][0],geo[count_i][1],geo[count_i][2]))
    return 

# Description: Simple wrapper function for writing a mol (V2000) file
#
# Inputs      name:     string holding the filename of the output
#             elements: list of element types (list of strings)
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#             adj_mat:  NxN array holding the molecular graph
#
# Returns     None
#
# new mol_write functions, can include radical info
def mol_write(name,elements,geo,adj_mat,q=0,append_opt=False):
    if len(elements) >= 1000:
        print( "ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return 

    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'

    # Parse the basename for the mol header
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]
    # Calls find_lewis from yarpecule
    e_lower = [_.lower() for _ in elements]
    bond_mat,bond_mat_scores = find_lewis(e_lower,adj_mat,q)
    fc = return_formals(bond_mat,e_lower)
    fc = fc[0]
    lones = []
    for i in range(len(bond_mat)):
        lones.append(np.diag(bond_mat[i]))
    bond_mat = bond_mat[0]
    keep_lone = [ [ count_j for count_j,j in enumerate(lone_electron) if j%2 != 0] for lone_electron in lones][0]
    chrg = len([i for i in fc if i != 0])
    with open(name,open_cond) as f:
        # Write the header
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))

        # Write the number of atoms and bonds
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0999 V2000\n".format(len(elements),int(np.sum(adj_mat/2.0))))

        # Write the geometry
        for count_i,i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2],i))

        # Write the bonds
        bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 and count_j > count_i ] 
        for i in bonds:

            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[i[0],i[1]])
                
            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1,i[1]+1,bond_order))
        #print(bond_order)
        # Write radical info if exist
        if len(keep_lone) > 0:
            if len(keep_lone) == 1:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}\n".format(1,keep_lone[0]+1,2))
            elif len(keep_lone) == 2:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}{:>4d}{:>4d}\n".format(2,keep_lone[0]+1,2,keep_lone[1]+1,2))
            else:
                print("Only support one/two radical containing compounds, radical info will be skip in the output mol file...")

        # Write charge info
        if chrg>0:
            if chrg == 1:
                charge = [i for i in fc if i != 0][0]
                charge = int(charge)              # charge was float, made it an integer
                fc_index = np.where(fc == charge) # using where() to get the indices of a numpy array
                #print(fc_index[0][0]+1)
                f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(1,fc_index[0][0]+1,charge))
                #f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(1,fc.index(charge)+1,charge))                
            else:
                info = "M  CHG{:>3d}".format(chrg)
                for count_c,charge in enumerate(fc):
                    charge = int(charge) # charge was float, made it an integer
                    if charge != 0: info += '{:>4d}{:>4d}'.format(count_c+1,charge) # changed 4d formatting to 4f, but didnt work
                info += '\n'
                f.write(info)
        f.write("M  END\n$$$$\n")

    return 

# Description: Simple wrapper function for grabbing the coordinates and
#              elements from an xyz file
#
# Inputs      input: string holding the filename of the xyz
# Returns     Elements: list of element types (list of strings)
#             Geometry: Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
def xyz_parse(input,read_types=False):

    # Commands for reading only the coordinates and the elements
    if read_types is False:
        
        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            if len(fields) > 4:
                                Atom_types[count] = fields[4]
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry,Atom_types

# Description: Parses the molecular charge from the comment line of the xyz file if present
#
# Inputs       input: string holding the filename of the xyz file. 
# Returns      q:     int or None
#
def parse_q(xyz):

    with open(xyz,'r') as f:
        for lc,lines in enumerate(f):
            if lc == 1:
                fields = lines.split()
                if "q" in fields:
                    q = int(float(fields[fields.index("q")+1]))
                else:
                    q = 0
                break
    return q

# Description: Checks if an array "a" is unique compared with a list of arrays "a_list"
#              at the first match False is returned.
def array_unique(a,a_list):
    for ind,i in enumerate(a_list):
        if np.array_equal(a,i):
            return False,ind
    return True,0

# Helper function to check_lewis and get_bonds that rolls the loop_list carbon elements
def reorder_list(loop_list,atomic_number):
    c_types = [ count_i for count_i,i in enumerate(loop_list) if atomic_number[i] == 6 ]
    others  = [ count_i for count_i,i in enumerate(loop_list) if atomic_number[i] != 6 ]
    if len(c_types) > 1:
        c_types = c_types + [c_types.pop(0)]
    return [ loop_list[i] for i in c_types+others ]

# Description:
# Rotate Point by an angle, theta, about the vector with an orientation of v1 passing through v2. 
# Performs counter-clockwise rotations (i.e., if the direction vector were pointing
# at the spectator, the rotations would appear counter-clockwise)
# For example, a 90 degree rotation of a 0,0,1 about the canonical 
# y-axis results in 1,0,0.
#
# Point: 1x3 array, coordinates to be rotated
# v1: 1x3 array, point the rotation passes through
# v2: 1x3 array, rotation direction vector
# theta: scalar, magnitude of the rotation (defined by default in degrees)
def axis_rot(Point,v1,v2,theta,mode='angle'):

    # Temporary variable for performing the transformation
    rotated=np.array([Point[0],Point[1],Point[2]])

    # If mode is set to 'angle' then theta needs to be converted to radians to be compatible with the
    # definition of the rotation vectors
    if mode == 'angle':
        theta = theta*np.pi/180.0

    # Rotation carried out using formulae defined here (11/22/13) http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/)
    # Adapted for the assumption that v1 is the direction vector and v2 is a point that v1 passes through
    a = v2[0]
    b = v2[1]
    c = v2[2]
    u = v1[0]
    v = v1[1]
    w = v1[2]
    L = u**2 + v**2 + w**2

    # Rotate Point
    x=rotated[0]
    y=rotated[1]
    z=rotated[2]

    # x-transformation
    rotated[0] = ( a * ( v**2 + w**2 ) - u*(b*v + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*x*np.cos(theta) + L**(0.5)*( -c*v + b*w - w*y + v*z )*np.sin(theta)

    # y-transformation
    rotated[1] = ( b * ( u**2 + w**2 ) - v*(a*u + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*y*np.cos(theta) + L**(0.5)*(  c*u - a*w + w*x - u*z )*np.sin(theta)

    # z-transformation
    rotated[2] = ( c * ( u**2 + v**2 ) - w*(a*u + b*v - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*z*np.cos(theta) + L**(0.5)*( -b*u + a*v - v*x + u*y )*np.sin(theta)

    rotated = rotated/L
    return rotated