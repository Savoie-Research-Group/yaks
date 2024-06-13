import sys,os,argparse,fnmatch
import pickle, re
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import NumRadicalElectrons
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import RDLogger 
import copy, time
from copy import deepcopy
from collections import defaultdict, Counter
from rdkit import rdBase

# all functions in taffi
sys.path.append('../utilities')
from taffi_functions import *

# Author: Michael Woulfe
""" Along with yaml_config.txt, script takes all reactions from yarp output folders listed in depot and scratch 
and compiles into a .yaml file, with no duplicate reactions, at which point a simple cantera script can simulate the
full network until steady state, giving the next node for YARP to explore off of.
"""
def main(argv):

    parser = argparse.ArgumentParser(description='Driver script for submitting YARP jobs with Gaussian engine.')

    #optional arguments                                             
    parser.add_argument('-c', dest='config', default='yaml_config.txt',
                        help = 'The program expects a configuration file from which to assign various run conditions. (default: yaml_config.txt in the current working directory)')
    parser.add_argument('-f', dest='folder', default='',
                        help = 'path to folder with yarp outputs)')
    parser.add_argument('-f2', dest='folder2', default='',
                        help = 'possible second folder with yarp outputs)')
    parser.add_argument('-fn', dest='filename', default='',
                        help = 'name the yaml file)')
    parser.add_argument('-input_folds', dest='input_folds', default='', help = 'list of folders to search in directory')

    print("parsing configuration directory...")
    args=parser.parse_args()
    c = parse_configuration(parser.parse_args())
    folder1 = args.folder
    folder2 = args.folder2
    file_name = args.filename
    input_folds = args.input_folds
    write_yaml_file(c, folder1, folder2, file_name, input_folds)

    return

def write_yaml_file(c, folder1 ='', folder2='', file_name = '', input_folds = ''):

    ###########################################
    ####        Parse Configuration        ####
    ###########################################

    start_t = time.time()
    input_list = []
    reac_dict = {}
    prod_dict = {}
    energy_dict = {}
    species = []

    # if specific folders are given, use them.  Else, default to the config file 
    if folder1:
        scr_path = folder1
        if input_folds: # if we give specific folders to search, only search those
            scr_folds = input_folds
        else: scr_folds = [f for f in os.listdir(scr_path) if os.path.isdir(os.path.join(scr_path, f))]
        # print(scr_path, scr_folds)
    elif c['scratch_folds'] == 'NA':
        scr_folds = ['NA']
        scr_path = c['calc_folder'] # give a random folder without anything useful
    else: 
        scr_path = c["scratch_path"]
        scr_folds = c["scratch_folds"]
        if type(scr_folds) == str: scr_folds = [scr_folds] # in case only one or zero folders are inputted
    if folder2:
        if folder2 == 'NA':
            dpt_folds = ['NA']
            dpt_path = c['calc_folder']
        else:
            dpt_path = folder2
            dpt_folds = [f for f in os.listdir(dpt_path) if os.path.isdir(os.path.join(dpt_path, f))]
    elif c['depot_folds'] == 'NA':
        dpt_folds = ['NA']
        dpt_path = c['calc_folder'] # give a random folder without anything useful
    else:
        dpt_path = c["depot_path"]
        dpt_folds = c["depot_folds"]
        if type(dpt_folds) == str: dpt_folds = [dpt_folds]
    

    ea_cutoff = float(c["ea_cutoff"])

    if c["rads"] == 'true' or c["rads"] == True: rads = True
    elif c["rads"] == "false" or c["rads"] == False: rads = False
    else: rads = False
    if c["rev_rxns"]=="true" or c["rev_rxns"] == True: rev_rxn = True
    else: rev_rxn = False

    print("\nParsing DFT Energy Dict")
    E_dict=parse_Energy(c["e_dict"])
    
    reactant_dict = c["reactant_dict"]
    smi2inchi = c["smi2inchi"]
    # load in reactant dictionaries
    print("Loading reactant dictionaries...")
    with open(reactant_dict,'rb') as f:
        reactant_dict = pickle.load(f)
    print("Loading Smile to Inchi Dict...")
    if os.path.isfile(smi2inchi):
        with open(smi2inchi,'rb') as f: 
            smi_inchi_dict = pickle.load(f)
    else: smi_inchi_dict = {}

    print(f"Time to load dicts: {time.time()-start_t}")

    #  IGNORE RDKIT Valence Warnings below
    RDLogger.DisableLog('rdApp.*') # comment out to see rdkit valence warnings.  Primarily for CWAs/N weird valences
    # blocker = rdBase.BlockLogs()


    ###############################################
    #### Determine Reactant Smiles/Inchi-Index ####
    ###############################################

    # returns input_list of full reactant Inchis
    get_reac_inchi(scr_folds, scr_path, input_list)  
    get_reac_inchi(dpt_folds, dpt_path, input_list)
    
    # recovers reactant smiles and inchis
    reac_smiles = {}
    # print("input_list",input_list) #### debug
    for inchi_ind in input_list:
        if inchi_ind not in reactant_dict.keys() or "possible_products" not in reactant_dict[inchi_ind].keys():
            print(f"{inchi_ind} missing possible products, first run reaction enumeration for it...")
            input_list.remove(inchi_ind)
        else:
            prop  = reactant_dict[inchi_ind]["prop"]
            reac_smiles[inchi_ind]= clean_reac(prop["smiles"].split('.')) # put reactants into temp dictionary, reformatted further down
        # print(f'Inchi-Index: {inchi_ind},  Smiles: {reac_smiles[inchi_ind]}') # debugging

    # formats reactant info in dict
    for key, val in reac_smiles.items():
        # print(key, val)
        if len(key.split('-')) == 1: # unimolecular
            reac_dict[key] = clean_reac(val[0])
            species.append(clean_reac(val[0]))
        if len(key.split('-')) == 2: # bimolecular
            new_key = f"{key.split('-')[0][:5]}-{key.split('-')[1][:5]}"
            reac_dict[new_key] = f'{val[0]} + {val[1]}'
            species.append(clean_reac(val[0]))
            species.append(clean_reac(val[1]))
            
    #############################################
    ####    PARSE report.txt/IRC-report.txt  ####
    #############################################
    print("\nParsing scratch report.txt...")
    parse_report(scr_folds, scr_path, prod_dict, energy_dict, species, ea_cutoff, smi_inchi_dict)
    print("\nParsing depot report.txt...")
    parse_report(dpt_folds, dpt_path, prod_dict, energy_dict, species, ea_cutoff, smi_inchi_dict)

    print("\nParsing scratch IRC-report.txt...")
    parse_IRC(scr_folds,scr_path, prod_dict, reac_dict, energy_dict, species, ea_cutoff, smi_inchi_dict)
    print("\nParsing depot IRC-report.txt...")
    parse_IRC(dpt_folds,dpt_path, prod_dict, reac_dict, energy_dict, species, ea_cutoff, smi_inchi_dict)

    # Keep lowest energy rxn if there are duplicates
    print("\n Removing duplicate reactions...")
    species = rm_duplicates(species) 
    elements = extract_elements(species) # get elements for, rev rxn calcs and .yaml writing
    rm_dup_rxns(prod_dict, energy_dict) 

    ###### debugging ######
    # print(f'len(species): {len(species)}\n {species}')
    # print(f'len(prod_dict): {len(prod_dict)}\n {prod_dict.items()}')
    # print(prod_dict['JJIUCEJQJXNMHV_102_0_4'])
    # print(prod_dict['JJIUCEJQJXNMHV_102_0_4'][1])
    # print(f'len(reac_dict): {len(reac_dict)}\n {reac_dict}')
    # print(f'len(energy_dict): {len(energy_dict)}\n {energy_dict}')
    # print(input_list)
    # print(smi_inchi_dict)
    # print(f"E_dict {E_dict}")

    # Calculate reverse rxns
    if rev_rxn: 
        exp_E_dict = {key: E_dict.get(key) for key in input_list} # only calculate rev rxns of explored nodes
        # exp_E_dict = E_dict # if you want to calculate all possible rev rxns
        # print(len(exp_E_dict))
        if exp_E_dict.get('XLYOFNOQVPJJNP') == None: # add water if not explictly explored upon
            exp_E_dict["XLYOFNOQVPJJNP"]={'E_0': -76.439454, 'H': -76.435674, 'F': -76.457099, 'SPE': -76.460631} 
        prod_dict, reac_dict, energy_dict, smi_inchi_dict = add_rev_rxns(exp_E_dict, prod_dict, reac_dict, energy_dict, smi_inchi_dict, ea_cutoff, elements)

        print("\n Removing duplicate reactions...")
        # Keep lowest energy rxn if there are duplicates
        # print(prod_dict)
        # print(energy_dict)  
        species = rm_duplicates(species)
        rm_dup_rxns(prod_dict, energy_dict)
        # print(prod_dict)
        # print(reac_dict)
    with open(smi2inchi,'wb') as f: pickle.dump(smi_inchi_dict, f, protocol=pickle.HIGHEST_PROTOCOL) # dump smi2inchi.dict to reuse
    
    ###########################################
    ####           WRITE YAML FILE         ####
    ###########################################
    # create folder
    if os.path.isdir(c["output_path"]) is False: os.mkdir(c["output_path"])

    # get a name for the yaml if it's not dictated
    if 'file_name' not in locals():
        file_name = c['calc_folder'].split('/')[-1]

    output_path = c["output_path"]
    print(f"\nWriting {file_name}.yaml...")

    # writes .yaml file header
    f = open(f'{output_path}/{file_name}.yaml', 'w')
    f.write(f"units:{{activation-energy: {c['energy_units']}}}\n\n"\
        "phases:\n"\
       f"- name: {file_name}\n"\
       f"  thermo: {c['eos']}\n"\
       f"  elements: {str(elements).translate({39: None})}\n"\
    #    f"  species: {species}\n"\
       f"  kinetics: {c['kinetics']}\n"\
       f"  reactions: {c['reactions']}\n")

    f = open(f'{output_path}/{file_name}.yaml', 'a')

    # write initial composition of system (X: or Y:)
    if c["rule"].lower() == "final_mass": mass_based = True
    else: mass_based = False

    mol_frac = {}
    mass_var = c["initial_reactants"]
    # can pull smiles from ERS_0.out to ensure it's the one ERS_enumeration actually uses
    
    # determine if user inputted smiles is same as yarp outputted smiles
    if type(mass_var) == str:
        # if only 1 input species, must contain all mass of system
        input_species = mass_var.split(":")[0]
        mol_frac[input_species] = mass_var.split(":")[1]
        for specie in species:
            if are_smiles_equal(specie, input_species) == True:
                # replace user supplied input with yaml output file
                input_species   = specie
                # add new k,v to mol_frac dict
                mol_frac[specie] = mol_frac[input_species]
    # replace multiple initial reactant inputs as well
    elif type(mass_var) == list:
        input_species = []
        for i in mass_var:
            mol_frac[i.split(":")[0]]=i.split(":")[1]
            input_species.append(i.split(":")[0])
        for cou, i in enumerate(input_species):
            for specie in species:
                if are_smiles_equal(specie, i):
                    input_species[cou] = specie
                    # make new key to correspond to proper smiles
                    mol_frac[specie] = mol_frac[i]


    # write species section
    if mass_based == True: f.write(f"  state:\n    Y: {{")
    else: f.write(f"  state:\n    X: {{")
    for cou, specie in enumerate(species):
        # if there is only a single input reactant
        if specie in mol_frac:
            if cou == len(species)-1: f.write(f"'{specie}': {mol_frac[specie]}")
            else: f.write(f"'{specie}': {mol_frac[specie]}, ")
        # else:
        #     if cou == len(species)-1: f.write(f"'{specie}': 0")
        #     else: f.write(f"'{specie}': 0, ")
    
    temp = int(c['temp'])
    f.write(f"}}\n")
    f.write(f"    T: {temp} K\n"\
            f"    P: {c['pres']} {c['pres_units']}\n\nspecies:\n")

    # section below writes out individual species elements (just need number of elements of each molecule)
    # print(f'len(species): {len(species)}')
    # print(f'species: {species}')
    for count, specie in enumerate(species): # write SPECIES section
        num = count_atoms(specie, elements)
        f.write(f"- name: '{specie}'\n  composition: {{")
        for i in elements: f.write(f"{i}: {str(num[i])}, ")
        f.write(f"}}\n  thermo:\n    model: {c['model_type']}\n    T0: {temp} K\n  equation-of-state: {c['eos']}\n\n")

    # TST equation calc Kb*T/h exp(-Ea/(R*T)), A = Kb*T/h
    K_b = 1.38*10**-23
    h = 6.626*10**-34
    A_value = int(K_b * temp / h) 
    A_units = 's^-1'

    f.write(f"reactions:\n") # write RXN section
    for k, p_list in prod_dict.items():
        r_inchi = k.split('_')[0]
        r_smile = reac_dict[r_inchi]
        r_list = r_smile.split(' + ')

        if fnmatch.fnmatch(k, '*cat*') and "O" in p_list:
            p_list.remove("O")
        

        # generate product smiles
        if len(p_list) == 1: p_smile = f"{p_list[0]}"
        if len(p_list) == 2: p_smile = f"{p_list[0]} + {p_list[1]}"
        if len(p_list) == 3: p_smile = f"{p_list[0]} + {p_list[1]} + {p_list[2]}"
        if len(p_list) == 4: p_smile = f"{p_list[0]} + {p_list[1]} + {p_list[2]} + {p_list[3]}"

        # determine units
        if len(r_inchi.split('-')) == 1: A_units = 's^-1' #unimolecular
        elif len(r_inchi.split('-')) == 2: A_units = 'm^3/mol/s' #bimolecular
        elif len(r_inchi.split('-')) == 3: A_units = 'm^6/mol^2/s' #trimolecular
        elif len(r_inchi.split('-')) ==4: A_units = 'm^9/mol^3/s' # quad
    
        # print(r_inchi, r_smile) # react_dict key, value
        # ensure reactants and products are different
        if compare_list(r_list, p_list) == False: 
            # sum of each element on both sides of equations
            prod_sum = dsum([val for val in count_atoms(p_list,elements).values()])
            reac_sum = dsum([val for val in count_atoms(r_list,elements).values()])

            # closed shell reactions
            if prod_sum == reac_sum and fnmatch.fnmatch(p_smile, "*[[]*[]]*") == False: # closed shell reactions
                f.write(f"- equation: '{r_smile} => {p_smile}'\n  rate-constant: {{A: {A_value} {A_units}, b: 0, Ea: {energy_dict[k]} {c['energy_units']}}} # eq {k}\n\n")
            
            # closed shell but includes H2
            elif prod_sum == reac_sum and '[H][H]' in p_smile: 
                f.write(f"- equation: '{r_smile} => {p_smile}'\n  rate-constant: {{A: {A_value} {A_units}, b: 0, Ea: {energy_dict[k]} {c['energy_units']}}} # eq {k}\n\n")
            
            # radical equations if flag is turned on or off
            elif prod_sum == reac_sum and fnmatch.fnmatch(p_smile, "*[[]*[]]*"): 
                if rads == True: f.write(f"- equation: '{r_smile} => {p_smile}'\n  rate-constant: {{A: {A_value} {A_units}, b: 0, Ea: {energy_dict[k]} {c['energy_units']}}} # eq {k}\n\n")
                elif rads == False: f.write(f" # - equation: '{r_smile} => {p_smile}'\n #  rate-constant: {{A: {A_value} {A_units}, b: 0, Ea: {energy_dict[k]} {c['energy_units']}}} # radicals/cation/anion eq {k}\n\n")
            
            # exception for water catalyzed reactions
            elif prod_sum != reac_sum and fnmatch.fnmatch(k, '*cat*'):
                if len(r_inchi.split('-')) == 1: A_units = 'm^3/mol/s'    
                elif len(r_inchi.split('-')) ==2:  A_units = 'm^6/mol^2/s' 
                elif len(r_inchi.split('-')) == 3:  A_units = 'm^9/mol^3/s' 
                elif len(r_inchi.split('-')) ==4: A_units = 'm^12/mol^4/s' 
                if dsub(prod_sum, reac_sum) == {'H':2, 'O':1}: # exception for water cat combination
                    f.write(f"- equation: '{r_smile} + O => {p_smile}'\n  rate-constant: {{A: {A_value} {A_units}, b: 0, Ea: {energy_dict[k]} {c['energy_units']}}} # eq {k}\n\n")
            
            # a catch all statement
            elif prod_sum == reac_sum: f.write(f"- equation: '{r_smile} => {p_smile}'\n  rate-constant: {{A: {A_value} {A_units}, b: 0, Ea: {energy_dict[k]} {c['energy_units']}}} # eq {k}\n\n")
            
            # write all the bad reactions so user can verify and trace error to source
            elif prod_sum != reac_sum:  f.write(f"# - equation: '{r_smile} => {p_smile}'\n  # rate-constant: {{A: {A_value} {A_units}, b: 0, Ea: {energy_dict[k]} {c['energy_units']}}} # improper amount of atoms on each side of eq {k} Atom Error (R-P): {dsub(prod_sum, reac_sum)}\n\n")
            
            # write all the bad reactions so user can verify and trace error to source
            else:  f.write(f"# - equation: '{r_smile} => {p_smile}'\n  # rate-constant: {{A: {A_value} {A_units}, b: 0, Ea: {energy_dict[k]} {c['energy_units']}}} # Likely couldn't read number of atoms on each side of equation {k} Atom Error (R-P): {dsub(prod_sum, reac_sum)}\n\n")
    print(f"Total time of function is {time.time()-start_t} seconds.")
  
###########################################
####              FUNCTIONS            ####
###########################################

# Function for keeping tabs on the validity of the user supplied inputs
def parse_configuration(args):
    
    # Convert inputs to the proper data type
    if os.path.isfile(args.config) is False:
        print("ERROR in python_driver: the configuration file {} does not exist.".format(args.config))
        quit()
    
    # Process configuration file for keywords

    # keywords
    keywords = ["input_type","reactant_dict","smi2inchi","output_path","pygsm_path","e_dict","ff","apply_tcit","tcit_result","hf_cut","charge","unpair","batch","sched","restart_step","eps",\
                "select_conf","low-irc","dft-irc","criterion","memory","irc-image","stepsize","irc-model","dg_thresh","compute_product","level","nimage","add-tor","conv-tor","relax_end","low_wt","low_njobs","low_queue","low_procs",\
                "c_method","c_wt","c_njobs","c_nprocs","n_max","c_queue","add-joint","c_path","functional","basis","dispersion","solvation","high_wt","ppn","high_procs","high_njobs","high_queue","parallel",\
                "rads","rev_rxns","energy_units","eos","kinetics","reactions","transport","temp","pres","pres_units","model_type","file_name","initial_reactants","EA_cutoff","ERS_queue","ERS_walltime",\
                "calc_folder", "config_template", "scratch_path","scratch_folds","depot_path","depot_folds","time_step","steps","rule","reactor","num_nodes"]

    keywords = [ _.lower() for _ in keywords ]
    
    list_delimiters = [","]  # values containing any delimiters in this list will be split into lists based on the delimiter
    space_delimiters = ["&"] # values containing any delimiters in this list will be turned into strings with spaces replacing delimiters
    configs = { i:None for i in keywords }    

    with open(args.config,'r') as f:
        for lines in f:
            fields = lines.split()
            
            # Delete comments
            if "#" in fields:
                del fields[fields.index("#"):]

            # Parse keywords
            l_fields = [ _.lower() for _ in fields ] 
            for i in keywords:
                if i in l_fields:
                    # Parse keyword value pair
                    ind = l_fields.index(i) + 1
                    if len(fields) >= ind + 1:
                        configs[i] = fields[ind]

                        # Handle delimiter parsing of lists
                        for j in space_delimiters:
                            if j in configs[i]:
                                configs[i] = " ".join([ _ for _ in configs[i].split(j) ])
                        for j in list_delimiters:
                            if j in configs[i]:
                                configs[i] = configs[i].split(j)
                                break
                        # Deal with -
                        #if '-' in configs[i]:
                        #    configs[i] = configs[i].replace('-',' ')
                                
                    # Break if keyword is encountered in a non-comment token without an argument
                    else:
                        print("ERROR in python_driver: enountered a keyword ({}) without an argument.".format(i))
                        quit()
    return configs

# creates dictionary of type of atoms and number for each molecule
# requires string or list input to dict or dict of dicts output
def count_atoms(smiles, elements): # want to keep elements so all species are labeled with all possible elements
    # print(smiles) # debug
    if isinstance(smiles, list):
        list_dict={}
        for count, val in enumerate(smiles):
            # converts smiles to rdkit object and then back to molecular formula
            mol = Chem.MolFromSmiles(val)
     
            if not mol: # smiles has improper valence and won't convert without sanitize flag being false
                mol = Chem.MolFromSmiles(val, sanitize = False)
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
            mol_form = rdMolDescriptors.CalcMolFormula(mol)
            
            # tokenizes chemical formula
            tokens = re.findall('[A-Z][a-z]?|\\d+|.', mol_form)
            num_dict = {}
            
            for i in elements: num_dict[i]=0
            # get formal charge for each species
            formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            num_dict['E'] = formal_charge
            
            for cou, v in enumerate(tokens):
                if cou == len(tokens)-1: # error catch for last value in token list if it's a chemical symbol
                    if v.isalpha(): num_dict[v] = 1
                else:
                    # if next token in chem formula is chemical symbol
                    if v.isalpha() and tokens[cou+1].isalpha():
                        num_dict[v] = 1
                    # if next token in chem formula is number
                    elif v.isalpha() and tokens[cou+1].isnumeric():
                        num_dict[v] = int(tokens[cou+1])
                    # if next token in chem formula is a charge
                    elif v.isalpha() and (tokens[cou+1] == '-' or tokens[cou+1] == '+'):
                        num_dict[v] = 1
            if val in list_dict:
                list_dict[f"{val}_{count}"] = num_dict
            else:
                list_dict[val] = num_dict
        return list_dict
    
    elif isinstance(smiles, str):
        # converts smiles to rdkit object and then back to molecule formula
        mol = Chem.MolFromSmiles(smiles)
        if not mol: # smiles has improper valence and won't convert to mol without sanitize = False
            mol = Chem.MolFromSmiles(smiles, sanitize = False)
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
        mol_form = rdMolDescriptors.CalcMolFormula(mol)
        # tokenizes chemical formula
        tokens = re.findall('[A-Z][a-z]?|\\d+|.', mol_form)
        num_dict = {}
        for i in elements: num_dict[i]=0
        # get formal charge for each species
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        num_dict['E'] = formal_charge
        
        for cou, v in enumerate(tokens):
            if cou == len(tokens)-1: # error catch for last value in token list if it's a chemical symbol
                if v.isalpha(): num_dict[v] = 1
            else:
                # if next token in chem formula is chemical symbol
                if v.isalpha() and tokens[cou+1].isalpha():
                    num_dict[v] = 1
                # if next token in chem formula is number
                elif v.isalpha() and tokens[cou+1].isnumeric():
                    num_dict[v] = int(tokens[cou+1])
                # if next token in chem formula is a charge
                elif v.isalpha() and (tokens[cou+1] == '-' or tokens[cou+1] == '+'):
                    num_dict[v] = 1
        return num_dict

# extract elements from smiles str/list of smiles strs
def extract_elements(smiles):

    # from rdkit.Chem.Descriptors import NumRadicalElectrons
    # from rdkit import Chem
    # import re

    all_elements = set()
    if isinstance(smiles, list):
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile, sanitize = False)
            if mol:  # Checks if the molecule was parsed correctly
                for atom in mol.GetAtoms():
                    all_elements.add(atom.GetSymbol())
    elif isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles, sanitize = False)
        if mol:  # Checks if the molecule was parsed correctly
            for atom in mol.GetAtoms():
                all_elements.add(atom.GetSymbol())
        
    all_elements.update('H')
    all_elements.update('E')
    all_elements = list(all_elements)
    return all_elements

# remove unwanted text from list of smiles strings 
def clean_smiles(smiles):
    rep = {"/":"", "\\":""}
    ele = extract_elements(smiles)
    for i in ele:
        rep[f"[{i}@H]"] = f"{i}"
        rep[f"[{i}@@H]"]= f"{i}"
        rep[f"[{i}@]"]  = f"{i}"
        rep[f"[{i}@@]"] = f"{i}"
        rep[f"[{i}@+]"]  = f"[{i}+]"
        rep[f"[{i}@-]"]  = f"[{i}-]"

    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile('|'.join(rep.keys()))
    for cou, i in enumerate(smiles):
        smiles[cou] = pattern.sub(lambda m: rep[re.escape(m.group(0))], i)
    return smiles

# remove unwanted text from smiles string
# has becomwe unwieldly and haggard for N 4 valence exception and handling radicals
def cln_sm(smiles):
    if isinstance(smiles, str):
        # 1st part runs standard cleaning routine
        mol = Chem.MolFromSmiles(smiles)
        if mol: rads = NumRadicalElectrons(mol) # find number of electrons if it's not a valence error
        rep = {"/":"", "\\":""}
        ele = extract_elements(smiles)
        for i in ele:
            rep[f"[{i}@H]"] = f"{i}"
            rep[f"[{i}@@H]"]= f"{i}"
            rep[f"[{i}@]"]  = f"{i}"
            rep[f"[{i}@@]"] = f"{i}"
            rep[f"[{i}@+]"]  = f"[{i}+]"
            rep[f"[{i}@-]"]  = f"[{i}-]"

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile('|'.join(rep.keys()))
        new_smile = pattern.sub(lambda m: rep[re.escape(m.group(0))], smiles)
        # 2nd part confirms # of rads stays the same and if not, alters cleaning protocol
        if mol: 
            if rads == NumRadicalElectrons(Chem.MolFromSmiles(new_smile)): # if number of radicals on molecule stays the same after cleaning
                return new_smile
            else: # redo with different conditions
                # print(smile)
                for i in ele:
                    rep[f"[{i}@H]"] = f"{i}"
                    rep[f"[{i}@@H]"]= f"{i}"
                    rep[f"[{i}@]"]  = f"[{i}]" # try keeping radical on chiral atom
                    rep[f"[{i}@@]"] = f"[{i}]" # same
                    rep[f"[{i}@+]"]  = f"[{i}+]"
                    rep[f"[{i}@-]"]  = f"[{i}-]"
                rep = dict((re.escape(k), v) for k, v in rep.items())
                pattern = re.compile('|'.join(rep.keys()))
                new_smile = pattern.sub(lambda m: rep[re.escape(m.group(0))], smiles)
                if rads == NumRadicalElectrons(Chem.MolFromSmiles(smiles)): # if number of radicals on molecule stays the same after cleaning
                    return new_smile
        else: return new_smile # for valence issues (on N)

    if isinstance(smiles, list):
        new_smiles_list =[]
        for smile in smiles:
        # 1st part runs standard cleaning routine
            mol = Chem.MolFromSmiles(smile)
            if mol: rads = NumRadicalElectrons(mol) # find number of electrons if it's not a valence error
            rep = {"/":"", "\\":""}
            ele = extract_elements(smile)
            for i in ele:
                rep[f"[{i}@H]"] = f"{i}"
                rep[f"[{i}@@H]"]= f"{i}"
                rep[f"[{i}@]"]  = f"{i}"
                rep[f"[{i}@@]"] = f"{i}"
                rep[f"[{i}@+]"]  = f"[{i}+]"
                rep[f"[{i}@-]"]  = f"[{i}-]"

            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile('|'.join(rep.keys()))
            new_smile = pattern.sub(lambda m: rep[re.escape(m.group(0))], smile)
            # 2nd part confirms # of rads stays the same and if not, alters cleaning protocol
            if mol:
                # if number of radicals on molecule stays the same after cleaning 
                if rads == NumRadicalElectrons(Chem.MolFromSmiles(new_smile)): 
                    new_smiles_list.append(new_smile)
                else: # redo with different conditions
                    # print(smile)
                    for i in ele:
                        rep[f"[{i}@H]"] = f"{i}"
                        rep[f"[{i}@@H]"]= f"{i}"
                        rep[f"[{i}@]"]  = f"[{i}]" # try keeping radical on chiral atom
                        rep[f"[{i}@@]"] = f"[{i}]" # same
                        rep[f"[{i}@+]"]  = f"[{i}+]"
                        rep[f"[{i}@-]"]  = f"[{i}-]"
                    rep = dict((re.escape(k), v) for k, v in rep.items())
                    pattern = re.compile('|'.join(rep.keys()))
                    new_smile = pattern.sub(lambda m: rep[re.escape(m.group(0))], smile)
                    if rads == NumRadicalElectrons(Chem.MolFromSmiles(smile)): # if number of radicals on molecule stays the same after cleaning
                        new_smiles_list.append(new_smile)
            else: new_smiles_list.append(new_smile) # for valence issues (on N)
        return new_smiles_list

# for radicals/cations/anions.  While ERS_enumeration runs more smoothly with anions/cations explicitly labeled, cantera doesn't allow it
def clean_reac(smile):
    if isinstance(smile, str):
        # print("string")
        rep = {"/":"", "\\":""}
        ele = extract_elements(smile)
        for i in ele:
            rep[f"[{i}@H]"] = f"{i}"
            rep[f"[{i}@@H]"]= f"{i}"
            rep[f"[{i}@]"]  = f"{i}"
            rep[f"[{i}@@]"] = f"{i}"
            rep[f"[{i}@+]"]  = f"[{i}]"
            rep[f"[{i}@@+]"]  = f"[{i}]"
            rep[f"[{i}@-]"]  = f"[{i}]"
            rep[f"[{i}@@-]"]  = f"[{i}]"
            rep[f"[{i}+]"]  = f"[{i}]"
            rep[f"[{i}-]"]  = f"[{i}]"
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile('|'.join(rep.keys()))
        new_smile = pattern.sub(lambda m: rep[re.escape(m.group(0))], smile)
        return new_smile
    
    elif isinstance(smile, list):
        # print("list")
        smi_list = []
        for sm in smile:
            rep = {"/":"", "\\":""}
            ele = extract_elements(sm)
            for i in ele:
                rep[f"[{i}@H]"] = f"{i}"
                rep[f"[{i}@@H]"]= f"{i}"
                rep[f"[{i}@]"]  = f"{i}"
                rep[f"[{i}@@]"] = f"{i}"
                rep[f"[{i}@+]"]  = f"[{i}]"
                rep[f"[{i}@@+]"]  = f"[{i}]"
                rep[f"[{i}@-]"]  = f"[{i}]"
                rep[f"[{i}@@-]"]  = f"[{i}]"
                rep[f"[{i}+]"]  = f"[{i}]"
                rep[f"[{i}-]"]  = f"[{i}]"
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile('|'.join(rep.keys()))
            new_smile = pattern.sub(lambda m: rep[re.escape(m.group(0))], sm)
            smi_list.append(new_smile)

        return smi_list

# compares two lists for same elements
def compare_list(l1,l2):
    return sorted(l1) == sorted(l2)

# sum/sub dicts together or return single dict
def dsum(list_of_dicts):
    if isinstance(list_of_dicts, list):
        ret = defaultdict(int)
        for d in list_of_dicts:
            for k, v in d.items():
                ret[k] += v
        return dict(ret)
    elif isinstance(list_of_dicts, dict):
        return list_of_dicts

def dsub(x, y): 
    z = dict(Counter(x)-Counter(y))
    return z

# determines if two distinct smiles strings represent the same molecule
def are_smiles_equal(smiles1, smiles2):
    # Convert SMILES strings to RDKit molecules
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Check if molecules are valid
    if mol1 and mol2:

        # Generate canonical smiles to compare
        canonical_smiles1 = Chem.MolToSmiles(mol1, isomericSmiles=False, canonical=True)
        canonical_smiles2 = Chem.MolToSmiles(mol2, isomericSmiles=False, canonical=True)

        # Use canonical SMILES for comparison
        return canonical_smiles1 == canonical_smiles2

    # if molecules are invlad
    elif mol1 is None or mol2 is None:
        # but smiles strings are exact same
        if smiles1==smiles2:
            return True
        else: return False

# removes duplicate entries from list
def rm_duplicates(species):
    species = list(dict.fromkeys(species))
    return species

# purpose is to find duplicate rxns and only include minimum energy dupliate
def rm_dup_rxns(prod_dict,energy_dict):

    # list to track water cat indices we didn't remove a water from
    no_remove = []
    # remove water from water catalyzed products (ensures water catalyzed and normal reactions are treated the same)
    for key, p_list in prod_dict.items():
        # if key shows it's water catalyzed and water in product
        if fnmatch.fnmatch(key, '*cat*') and "O" in p_list:
            prod_dict[key].remove("O")
        # elif just water catalyzed, but no water in product
        elif fnmatch.fnmatch(key, '*cat*'): 
            no_remove.append(key)
    
    rxn_list = []
    # create list of lists of list
    # list of list[0] being products, list[1] being inchi keys
    for k, v in prod_dict.items():
        must_insert = True
        for val in rxn_list:
            if compare_list(val[0],v):
                val[1].append(k)
                must_insert = False
                break
        if must_insert: rxn_list.append([v, [k]])

    # print("rxn_list:", rxn_list)
    # create list of duplicate products
    dupls = []
    for i in rxn_list:
        if len(i[1]) >=2:
            dupls.append(i[1])
    # splits dupls list further to separate keys with different reactants 
    split_dupls = []
    for cou, lis in enumerate(dupls):
        temp = {} 
        for val in lis:
            temp.setdefault(val[:11], []).append(val) # separates each list into a dict based on different reactants
        temp_list = list(temp.values()) # transforms dict into list of lists
        split_dupls.extend(temp_list) # creates final split list

    # finds minimum energy of duplicate rxns
    save_keys = []
    for cou, lis in enumerate(split_dupls):
        temp = {}
        for i in lis:
            temp[i]=(energy_dict[i])    
        save_keys.append(min(temp, key=temp.get)) # gets key of minimum energy value

    # removes keys to save from dupls list
    for cou, lis in enumerate(split_dupls):
        for i in save_keys:
            if i in lis:
                lis.remove(i)

    # flattens split_dupls into a list (vice list of lists)
    flat_dupls = [item for sublist in split_dupls for item in sublist]

    # removes duplicate keys from prod_dict (2/3 in test case)
    for k in flat_dupls:
        prod_dict.pop(k, None)

    # add water back to water catalyzed products
    for key in list(prod_dict.keys()):
        if fnmatch.fnmatch(key, "*cata*") and key not in no_remove:
            prod_dict[key].append("O")

    duplicates = find_niche_duplicates(prod_dict)
    # check for exception: water catalyzed hydration rxn and reverse of a dehydration rxn (different labels)
    for k, v in duplicates.items():
        dupl_set = set()
        for cou, val in enumerate(v): # v is list of rxn indexes
            # look for bimolecular reaction with water
            if '-XLYOF' in val: 
                other_inchi = val.split("_")[0].split("-")[0]
                dupl_set.add(val)
                # check other keys for other inchi and wc rxn
                for cou, val in enumerate(v):
                    if val[:5] == other_inchi and 'cata' in val:
                        dupl_set.add(val)
                        
            elif 'XLYOF-' in val: 
                other_inchi = val.split("_")[0].split("-")[1]
                dupl_set.add(val)
                # check other keys for other inchi and wc rxn
                for cou, val in enumerate(v):
                    if val[:5] == other_inchi and 'cata' in val:
                        dupl_set.add(val)
        
        temp, keep = {}, []
        # get energies
        for i in dupl_set: temp[i]=(energy_dict[i])
        # if dupl isn't empty, get min value key
        if dupl_set: min_value_key = min(temp, key=temp.get)
        # add rest of keys to a list
        other_keys = [key for key in temp if key != min_value_key]
        # remove other keys from prod_dict
        for i in other_keys: prod_dict.pop(i)
    
    return prod_dict

# function to find duplicate values in a dictionary (for check at end of rm_dup_rxns)
def find_niche_duplicates(input_dict):
    # This dictionary will map the normalized values (as tuples) to a list of keys that have that value
    poss_dups = {}
    
    for key, value_list in input_dict.items():
        # Normalize the list by sorting it and converting it to a tuple for immutability
        normalized_value = tuple(sorted(value_list))
        
        # Append the key to the list of keys for this normalized value
        if normalized_value in poss_dups:
            poss_dups[normalized_value].append(key)
        else:
            poss_dups[normalized_value] = [key]
    
    # Filter to only keep the values that occurred more than once
    duplicates = {value: keys for value, keys in poss_dups.items() if len(keys) > 1}
    
    return duplicates

# parses report.txt into dictionary  to get reaction info
def parse_report(folds, path, prod_dict, energy_dict, species, ea_cutoff, smi_inchi_dict):
    if folds == ['NA']: # break for no entry
        return 
    for count, fold in enumerate(folds):
        # print(fold)
        for f in os.listdir(f'{path}/{fold}'):   
            if f == 'report.txt':
                with open(f'{path}/{fold}/report.txt', 'r') as g:
                    for lines in g:
                        line = lines.split()
                        if len(line) <4 or fnmatch.fnmatch(line[0], '#*'): 
                            pass # skip empty and 1st line 
                        elif (float(line[3]) < int(ea_cutoff)) and (float(line[3])>0):
                            energy_dict[line[0]] = line[3] # adds rxn act energy to dict
                            prods = line[1].split('.')
                            for cou, val in enumerate(prods): 
                                species.append(cln_sm(val)) # adds prods to species list
                                # print(cln_sm(val)) #debug
                                smi_inchi_dict, inchi_list = smiles_to_inchi(cln_sm(prods),smi_inchi_dict)
                                # prod_dict[line[0]] = inchi_list # keeps inchis (different smiles correspond to same inchi)
                                prod_dict[line[0]] = cln_sm(prods) # keeps cleaned smiles
                      

    return prod_dict, energy_dict, species, smi_inchi_dict

# parses IRC-report to find all rxns and exclude meaningless ones (built to handle up to 5 products)
def parse_IRC(folds, path, prod_dict, reac_dict, energy_dict,species, ea_cutoff, smi_inchi_dict):
    if folds == ['NA']: # break for no entry
        return 
    for count, fold in enumerate(folds):
        # print(fold)
        for f in os.listdir(f'{path}/{fold}'):
        #  IRC data parser using IRC-record.txt
            if f == 'IRC-result':
                with open(f'{path}/{fold}/IRC-result/IRC-record.txt', 'r') as h:
                    for lines in h:
                        line = lines.split()
                        if len(line)==0: pass # skip empty lines
                        elif line[-1]=='barrier': pass # skip 1st line                        
                        elif fnmatch.fnmatch(line[0], '#*'): pass # skip commented out lines
                        elif (float(line[-1])<int(ea_cutoff)) and (float(line[-1])>0):
                            energy_dict[line[0]] = line[-1] # adds rxn act energy to dict
                            prod1 = cln_sm(line[1].split('.'))
                            prod2 = cln_sm(line[2].split('.'))
                            r_inchi = line[0].split('_')[0] 

                            smi_inchi_dict, inchi_list1 = smiles_to_inchi(prod1, smi_inchi_dict) # create lists of inchis/ update smi_inchi_dict
                            smi_inchi_dict, inchi_list2 = smiles_to_inchi(prod2, smi_inchi_dict)
                            species.extend(cln_sm(prod1)) # add smiles to species list
                            species.extend(cln_sm(prod2))
                            # print(reac_dict)
                            reac_smile = reac_dict[r_inchi] # determine reactant smiles for line
                            # bimolecular
                            if len(reac_smile.split(' + '))>=2: 
                                smi_inchi_dict, reac_inchi_list = smiles_to_inchi(reac_smile.split(' + '), smi_inchi_dict)
                                if (all(x in inchi_list1 for x in reac_inchi_list)) and (all(x in inchi_list2 for x in reac_inchi_list)): # both sides have both
                                    pass
                                # R node 1
                                elif all(x in inchi_list1 for x in reac_inchi_list):
                                    if len(prod1) == 2 or (len(prod1) == 3 and 'cata' in line[0] and 'O' in prod1): 
                                        prod_dict[line[0]] = prod2
                                # R node 2
                                elif all(x in inchi_list2 for x in reac_inchi_list):
                                    if len(prod2) == 2 or (len(prod2) == 3 and 'cata' in line[0] and 'O' in prod2): 
                                        prod_dict[line[0]] = prod1

                            # unimolecular
                            if len(reac_smile.split(' + '))==1:
                                if r_inchi in inchi_list1 and r_inchi in inchi_list2: # both sides have reactant
                                    pass
                                # R on node 1 side
                                elif r_inchi in inchi_list1:
                                    if len(prod1) == 1 or (len(prod1) == 2 and 'cata' in line[0] and 'O' in prod1):
                                        prod_dict[line[0]] = prod2
                                # R on node 2 side
                                elif r_inchi in inchi_list2: 
                                    if len(prod2) == 1 or (len(prod2) == 2 and 'cata' in line[0] and 'O' in prod2):
                                        prod_dict[line[0]] = prod1
    return prod_dict, energy_dict, species

# iterates through all folders listed in yaml_config to parse reactant data
def get_reac_inchi(folds, path, input_list): 
    if folds == ['NA']:
        return 
    for count, fold in enumerate(folds):
        # add reactants from this fold to input_list via inchikey
        if os.path.isdir(f'{path}/{fold}/opt-folder/') is True: # quick way to find inchi
            for f in os.listdir(f'{path}/{fold}/opt-folder/'): 
                if fnmatch.fnmatch(f, '*-opt.xyz'): # there's one file in yarp opt-folder that has the Inchikey as its name
                    pass # all folders have -opt.xyz whereas the only file has xyz
                elif fnmatch.fnmatch(f, '*cata*'): # conf gen results files opt-folder has only cata geometries
                    pass 
                else:
                    input_list.append(f.split('.')[0]) # adds inchikey to input_list
        elif os.path.isdir(f'{path}/{fold}/low_calc/') is True: # if fold has been cleaned
            for f in os.listdir(f'{path}/{fold}/low_calc/'): 
                if f.split('_')[0] not in input_list:
                    input_list.append(f.split('_')[0]) # adds inchikey to input_list
        elif len(fold.split('_')) == 4: # if folder hasn't been cleaned yet
            if fold.split('_')[0] not in input_list:
                print(fold)
                input_list.append(fold.split('_')[0])
        elif os.path.isdir(f'{path}/{fold}'): # wc exception
            for f in os.listdir(f'{path}/{fold}'):
                if len(f.split("-")[0].split("_")) == 3 and f != "input_files_conf":
                    inchi = f.split("-")[0].split("_")[0]
                    input_list.append(inchi)
        else: print(f"Error: {fold} not found")

    return input_list

# takes in list of smiles strings (or single str) and returns dicts of k:v smiles:inchi & inchi:smiles (list of possible smiles)
def smiles_to_inchi(coord_file, inchi_dict):
    if isinstance(coord_file, str):
        if cln_sm(coord_file) in inchi_dict: # if smile already in dict
            inchi_ind = inchi_dict[cln_sm(coord_file)]
        else: 
            mol = Chem.MolFromSmiles(coord_file)
            if not mol:
                mol = Chem.MolFromSmiles(coord_file, sanitize = False)
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
            inchikey = Chem.inchi.MolToInchiKey(mol)
            inchi_ind = inchikey.split('-')[0]
            inchi_dict[cln_sm(coord_file)]= inchi_ind
        return inchi_dict, inchi_ind

    # if input is list of smiles, returns list of inchis
    elif isinstance(coord_file, list):
        inchi_list = []
        for i in coord_file:
            if cln_sm(i) in inchi_dict:
                inchi_ind = inchi_dict[cln_sm(i)]
                inchi_list.append(inchi_ind)
            else:
                mol = Chem.MolFromSmiles(i)
                if not mol:
                    mol = Chem.MolFromSmiles(i, sanitize = False)
                    mol.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
                inchikey = Chem.inchi.MolToInchiKey(mol)
                inchi_ind = inchikey.split('-')[0]
                inchi_dict[cln_sm(i)]= inchi_ind
                inchi_list.append(inchi_ind)
        return inchi_dict, inchi_list

# Function to check and append DFT energies for reactants
def parse_Energy(db_files,E_dict={}):
    if os.path.exists(db_files) is False:
        E_dict={}
        return E_dict
    with open(db_files,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0: continue
            if len(fields) ==0: continue
            if len(fields) >= 4:
                inchi = fields[0][:14]
                if fields[0] not in E_dict.keys():
                    E_dict[inchi] = {}
                    E_dict[inchi]["E_0"]= float(fields[1])
                    E_dict[inchi]["H"]  = float(fields[2])
                    E_dict[inchi]["F"]  = float(fields[3])
                    E_dict[inchi]["SPE"]= float(fields[4])

    return E_dict

#  generates reverse rxns w/ energies of explored upon nodes (using DFT energies)
def add_rev_rxns(E_dict, prod_dict, reac_dict, energy_dict, smi_inchi_dict, ea_cutoff,elements):
    print("\nCalculating reverse reactions...")
    # copy dict to iterate over and update simultaneously
    prod_dict_copy = deepcopy(prod_dict)
    reac_dict_copy = deepcopy(reac_dict)
    # print(E_dict)
    # print(reac_dict)
    conv = 627.503 # hartree to kcal/mol

    for k, v in prod_dict_copy.items():
        for key, value in reac_dict_copy.items():
            reac_v = value.split(' + ')
            if k.split('_')[0] == key:
                prod_sum = dsum([val for val in count_atoms(v, elements).values()]) # sum of reac/prod sides of eq
                reac_sum = dsum([val for val in count_atoms(reac_v, elements).values()])
                if prod_sum == reac_sum:
                    if len(key.split('-')) == 1: # unimolecular reactant
                        if len(v)==1: # 1 product
                            smi_inchi_dict, products = smiles_to_inchi(v, smi_inchi_dict) # takes care of smiles already being in inchi 
                            p1 = products[0]
                            # print(p1,v,key, value) # debug
                            if p1 in E_dict:
                                delta_E_rxn = float(E_dict[p1]['F']) - float(E_dict[key]['F']) # Gibbs (Prod - Reac)
                                delta_G = delta_E_rxn*conv
                                act_E = float(energy_dict[k])
                                rev_act_E = act_E-delta_G 
                                # print(f"For RXN  : {value} --> {v} \nDelta E rxn: {delta_E_rxn}\nDelta G: {delta_G}\nAE: {act_E}\nrev AE: {rev_act_E}\n ")
                                # encode into dicts
                                if float(rev_act_E) > 0 and float(rev_act_E) < int(ea_cutoff): 
                                    if p1 not in reac_dict:
                                        reac_dict[p1] = f"{v[0]}"
                                    if p1 != key.split('_'): # don't copy rxns with same reac and prod
                                        prod_dict[f"{p1}_rev_{k}"] = reac_v # input new "product" as list of smiles
                                        energy_dict[f"{p1}_rev_{k}"] = format(rev_act_E, '.4f')

                        elif len(v)==2: # 2 products
                            smi_inchi_dict, products = smiles_to_inchi(v, smi_inchi_dict)
                            p1, p2 = products[0],products[1]
                            prods = sorted([p1, p2]) # inchis in alphabetical order

                            # print(p1,p2,v,key, value) # debug
                            if p1 in E_dict and p2 in E_dict:
                                delta_E_rxn = float(E_dict[p1]['F']) + float(E_dict[p2]['F']) - float(E_dict[key]['F']) # Gibbs (Prods - Reac)
                                delta_G = delta_E_rxn*conv
                                act_E = float(energy_dict[k])
                                rev_act_E = act_E-delta_G 
                            
                                # encode into dicts
                                if float(rev_act_E) > 0 and float(rev_act_E) < int(ea_cutoff):
                                    bi_inchi = f"{prods[0][:5]}-{prods[1][:5]}"
                                    if bi_inchi not in reac_dict:
                                        reac_dict[bi_inchi] = f"{v[0]} + {v[1]}"
                                    if bi_inchi != key.split('_')[0]: # don't copy rxns with same reac and prod
                                        prod_dict[f"{bi_inchi}_rev_{k}"] = reac_v
                                        energy_dict[f"{bi_inchi}_rev_{k}"] = format(rev_act_E, '.4f')
                
                        elif len(v)==3: # 3 products
                            smi_inchi_dict, products = smiles_to_inchi(v, smi_inchi_dict)
                            p1, p2, p3 = products[0],products[1],products[2]
                            prods = sorted([p1, p2, p3]) # trimol inchis in alphabetical order
                            if p1 in E_dict and p2 in E_dict and p3 in E_dict: 
                                delta_E_rxn = float(E_dict[p1]['F']) + float(E_dict[p2]['F']) + float(E_dict[p3]['F']) - float(E_dict[key]['F']) # Gibbs (Prods - Reac)
                                delta_G = delta_E_rxn*conv
                                act_E = float(energy_dict[k])
                                rev_act_E = act_E-delta_G

                                # encode into dicts
                                if float(rev_act_E) > 0 and float(rev_act_E) < int(ea_cutoff):
                                    bi_inchi = f"{prods[0][:5]}-{prods[1][:5]}-{prods[2][:5]}" # tri inchi, reuse pointer
                                    if bi_inchi not in reac_dict:
                                        reac_dict[bi_inchi] = f"{v[0]} + {v[1]} + {v[2]}"
                                    if bi_inchi != key.split('_')[0]: # don't copy rxns with same reac and prod
                                        prod_dict[f"{bi_inchi}_rev_{k}"] = reac_v
                                        energy_dict[f"{bi_inchi}_rev_{k}"] = format(rev_act_E, '.4f')

                        elif len(v)==4: # 4 products
                            smi_inchi_dict, products = smiles_to_inchi(v, smi_inchi_dict)
                            p1, p2, p3, p4 = products[0],products[1],products[2],products[3]
                            prods = sorted([p1, p2, p3, p4]) # quad inchis in alphabetical order
                            if p1 in E_dict and p2 in E_dict and p3 in E_dict and p4 in E_dict: 
                                delta_E_rxn = float(E_dict[p1]['F']) + float(E_dict[p2]['F']) + float(E_dict[p3]['F']) + float(E_dict[p4]['F']) - float(E_dict[key]['F']) # Gibbs (Prods - Reac)
                                delta_G = delta_E_rxn*conv
                                act_E = float(energy_dict[k])
                                rev_act_E = act_E-delta_G

                                # encode into dicts
                                if float(rev_act_E) > 0 and float(rev_act_E) < int(ea_cutoff):
                                    bi_inchi = f"{prods[0][:5]}-{prods[1][:5]}-{prods[2][:5]}-{prods[3][:5]}" # quad inchi, reuse pointer
                                    if bi_inchi not in reac_dict:
                                        reac_dict[bi_inchi] = f"{v[0]} + {v[1]} + {v[2]} + {v[3]}"
                                    if bi_inchi != key.split('_')[0]: # don't copy rxns with same reac and prod
                                        prod_dict[f"{bi_inchi}_rev_{k}"] = reac_v
                                        energy_dict[f"{bi_inchi}_rev_{k}"] = format(rev_act_E, '.4f') 

                    elif len(key.split('-'))==2: # bimolecular reactant
                        smi_inchi_dict, reactants = smiles_to_inchi(reac_v, smi_inchi_dict)
                        r1, r2 = reactants[0],reactants[1]
                        # print(r1, r2)
                        if r1 in E_dict and r2 in E_dict:  # in case you try a bimolecular exp without a uni first
                            if len(v)==1: # 1 product
                                smi_inchi_dict, products = smiles_to_inchi(v, smi_inchi_dict) 
                                p1 = products[0]

                                if p1 in E_dict:
                                    delta_E_rxn = float(E_dict[p1]['F']) - float(E_dict[r1]['F']) - float(E_dict[r2]['F']) # Gibbs (Prod - Reac)
                                    delta_G = delta_E_rxn*conv
                                    act_E = float(energy_dict[k])
                                    rev_act_E = act_E-delta_G 

                                    # encode into dicts
                                    if float(rev_act_E) > 0 and float(rev_act_E) < int(ea_cutoff):
                                        if p1 not in reac_dict:
                                            reac_dict[p1] = f"{v[0]}"
                                        if p1 != key.split('_')[0]: # don't copy rxns with same reac and prod
                                            prod_dict[f"{p1}_rev_{k}"] = reac_v
                                            energy_dict[f"{p1}_rev_{k}"] = format(rev_act_E, '.4f')

                            elif len(v)==2: # 2 products
                                smi_inchi_dict, products = smiles_to_inchi(v, smi_inchi_dict) 
                                p1,p2 = products[0],products[1]
                                prods = sorted([p1, p2]) # bimol inchis in alphabetical order
                                # print(p1, p2, prods)
                                if p1 in E_dict and p2 in E_dict:
                                    delta_E_rxn = float(E_dict[p1]['F']) + float(E_dict[p2]['F']) - float(E_dict[r1]['F']) - float(E_dict[r2]['F']) # Gibbs (Prods - Reac)
                                    delta_G = delta_E_rxn*conv
                                    act_E = float(energy_dict[k])
                                    rev_act_E = act_E-delta_G 
                                
                                    # encode into dicts
                                    if float(rev_act_E) > 0 and float(rev_act_E) < int(ea_cutoff):
                                        bi_inchi = f"{prods[0][:5]}-{prods[1][:5]}"
                                        if bi_inchi not in reac_dict:
                                            reac_dict[bi_inchi] = f"{v[0]} + {v[1]}"
                                        if bi_inchi != key.split('_')[0]: # don't copy rxns with same reac and prod
                                            prod_dict[f"{bi_inchi}_rev_{k}"] = reac_v
                                            energy_dict[f"{bi_inchi}_rev_{k}"] = format(rev_act_E, '.4f')
                    
                            elif len(v)==3: # 3 products
                                smi_inchi_dict, products = smiles_to_inchi(v, smi_inchi_dict) 
                                p1,p2,p3 = products[0],products[1],products[2]
                                prods = sorted([p1, p2, p3]) # trimol inchis in alphabetical order
                                if p1 in E_dict and p2 in E_dict and p3 in E_dict: 
                                    delta_E_rxn = float(E_dict[p1]['F']) + float(E_dict[p2]['F']) + float(E_dict[p3]['F']) - float(E_dict[r1]['F']) - float(E_dict[r2]['F']) # Gibbs (Prods - Reac)
                                    delta_G = delta_E_rxn*conv
                                    act_E = float(energy_dict[k])
                                    rev_act_E = act_E-delta_G

                                    # encode into dicts
                                    if float(rev_act_E) > 0 and float(rev_act_E) < int(ea_cutoff):
                                        bi_inchi = f"{prods[0][:5]}-{prods[1][:5]}-{prods[2][:5]}" # tri inchi, reuse pointer
                                        if bi_inchi not in reac_dict:
                                            reac_dict[bi_inchi] = f"{v[0]} + {v[1]} + {v[2]}"
                                        if bi_inchi != key.split('_')[0]: # don't copy rxns with same reac and prod
                                            prod_dict[f"{bi_inchi}_rev_{k}"] = reac_v
                                            energy_dict[f"{bi_inchi}_rev_{k}"] = format(rev_act_E, '.4f')

                            elif len(v)==4: # 4 products
                                smi_inchi_dict, products = smiles_to_inchi(v, smi_inchi_dict) 
                                p1,p2,p3,p4 = products[0],products[1],products[2],products[3]
                                prods = sorted([p1, p2, p3, p4]) # quad inchis in alphabetical order
                                if p1 in E_dict and p2 in E_dict and p3 in E_dict and p4 in E_dict:
                                    delta_E_rxn = float(E_dict[p1]['F']) + float(E_dict[p2]['F']) + float(E_dict[p3]['F']) + float(E_dict[p4]['F']) - float(E_dict[r1]['F']) - float(E_dict[r2]['F']) # Gibbs (Prods - Reac)
                                    delta_G = delta_E_rxn*conv
                                    act_E = float(energy_dict[k])
                                    rev_act_E = act_E-delta_G
                                    # encode into dicts
                                    if float(rev_act_E) > 0 and float(rev_act_E) < int(ea_cutoff):
                                        bi_inchi = f"{prods[0][:5]}-{prods[1][:5]}-{prods[2][:5]}-{prods[3][:5]}" # quad inchi, reuse pointer
                                        if bi_inchi not in reac_dict:
                                            reac_dict[bi_inchi] = f"{v[0]} + {v[1]} + {v[2]} + {v[3]}"
                                        if bi_inchi != key.split('_')[0]: # don't copy rxns with same reac and prod
                                            prod_dict[f"{bi_inchi}_rev_{k}"] = reac_v
                                            energy_dict[f"{bi_inchi}_rev_{k}"] = format(rev_act_E, '.4f')
                
    return prod_dict, reac_dict, energy_dict, smi_inchi_dict

if __name__ == "__main__":
    main(sys.argv[1:])