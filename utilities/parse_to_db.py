import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import pickle,json,csv
import numpy as np

# Load modules in same folder        
from xtb_functions import xtb_energy,xtb_geo_opt
from taffi_functions import *
from job_submit import *
from utility import *

def main():
    
    # pathway setting
    DFT_db        = '/depot/bsavoie/data/YARP/DFT-db/B3LYPD3_TZVP.db'
    target_folder = '/depot/bsavoie/data/YARP_database/Glucose/origin/C4_S1'
    TCIT_config   = '/home/zhao922/bin/Github/YARP/version2.0/ERS_enumeration/TCIT-config.txt'
    output_csv    = '/depot/bsavoie/data/YARP_database/Glucose/YARP_reactions.csv'
    output_path   = '/depot/bsavoie/data/YARP_database/Glucose/reaction_db'
    fields        = ['index', 'Rsmiles', 'Psmiles', 'DE', 'DG', 'DH']

    # load in energies
    E_dict = parse_Energy(DFT_db)

    # create a csv file
    if os.path.isfile(output_csv) is False:
        with open(output_csv, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)

    # parse intended channels
    reactions = {}
    with open(target_folder+'/report.txt','r') as f:
        for lc,lines in enumerate(f):
            if lc == 0: continue
            fields = lines.split()
            Rindex = fields[0]
 
            # obtain corresponding files
            input_xyz = target_folder + '/input_files_conf/{}.xyz'.format(Rindex)
            TS_file   = target_folder + '/TS-folder/{}-TS.out'.format(Rindex)
            TS_xyz    = target_folder + '/TS-folder/{}-TS.xyz'.format(Rindex)

            # parse geo info
            E,RG,PG,Radj_mat,Padj_mat = parse_input(input_xyz,return_adj=True)
            _,TS_G = xyz_parse(TS_xyz)
            
            # parse thermo data
            _,_,SPE,zero_E,H_298,F_298,_ = read_Gaussian_output(TS_file)

            # generate atom mapped smiles
            rsmiles_map,rsmiles = return_atommaped_smi(E,RG,Radj_mat),return_smi(E,RG,Radj_mat).split('.')
            psmiles_map,psmiles = return_atommaped_smi(E,PG,Padj_mat),return_smi(E,PG,Padj_mat).split('.')
            rinchikey,pinchikey = return_inchi(E,RG,Radj_mat),return_inchi(E,PG,Padj_mat)

            # compute thermo 
            DG = (F_298 - sum([E_dict[inchi[:14]]['F'] for inchi in rinchikey]))*627.5095
            DE = (SPE - sum([E_dict[inchi[:14]]['SPE'] for inchi in rinchikey]))*627.5095
            TCIT_dict = TCIT_prediction(rsmiles+psmiles,TCIT_config)
            try:
                DH = (sum([TCIT_dict[smi]["Hf_298"] for smi in psmiles]) - sum([TCIT_dict[smi]["Hf_298"] for smi in rsmiles]))/4.184
            except:
                DH = "TCIT fails"

            if len(rinchikey) > 1: rinchikey = '-'.join([inchi[:14] for inchi in rinchikey])
            else: rinchikey = rinchikey[0]

            if len(pinchikey) > 1: pinchikey = '-'.join([inchi[:14] for inchi in pinchikey])
            else: pinchikey = pinchikey[0]

            reaction_inchi = '-'.join(sorted([rinchikey,pinchikey]))
            
            # determine whether reaction_inchi exists in the reaction dictionary
            if reaction_inchi not in reactions.keys():
                reactions[reaction_inchi] = {'1':[Rindex,rsmiles_map,psmiles_map,DE,DG,DH,E,RG,PG,TS_G]}
                reactions[reaction_inchi]['energy'] = {DG:'1'}
            else:
                index = check_dup(reactions[reaction_inchi]['energy'].keys(),DG,thresh=0.5)
                if not index:
                    nind = len(reactions[reaction_inchi])
                    reactions[reaction_inchi][str(nind)] = [Rindex,rsmiles_map,psmiles_map,DE,DG,DH,E,RG,PG,TS_G]
                    reactions[reaction_inchi]['energy'][DG]=str(nind)

    for reaction_inchi,reaction in reactions.items():
        for conf_ind,reaction_info in reaction.items():
            if conf_ind == 'energy': continue
            # unpack info
            Rindex,rsmiles_map,psmiles_map,DE,DG,DH,E,RG,PG,TS_G = reaction_info
            
            # write into csv 
            with open(output_csv, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([Rindex,rsmiles_map,psmiles_map,DE,DG,DH])
        
            # write reaction into a xyz file
            out_xyz = '{}/{}.xyz'.format(output_path,Rindex)
            xyz_write(out_xyz,E,RG)
            xyz_write(out_xyz,E,PG,append_opt=True)
            xyz_write(out_xyz,E,TS_G,append_opt=True,comment='DE: {}  DG: {} DH: {} (kcal/mol)'.format(DE, DG, DH))
            
    # parse intended channels
    reactions = {}
    with open(target_folder+'/IRC-result/IRC-record.txt','r') as f:
        
        for lc,lines in enumerate(f):
            if lc == 0: continue
            fields = lines.split()
            Rindex = fields[0]

            # locate reaction, TS files
            IRC_xyz   = target_folder + '/IRC-result/{}-IRC.xyz'.format(Rindex)
            TS_file   = target_folder + '/TS-folder/{}-TS.out'.format(Rindex)
            TS_xyz    = target_folder + '/TS-folder/{}-TS.xyz'.format(Rindex)
            input_xyz = target_folder + '/input_files_conf/{}.xyz'.format(Rindex)

            # parse target reaction info
            target_E,target_RG,_,target_Radj,_ = parse_input(input_xyz,return_adj=True)

            # parse thermo data
            _,_,SPE,zero_E,H_298,F_298,_ = read_Gaussian_output(TS_file)

            # parse IRC info
            E,G1,G2,adj_mat_1,adj_mat_2 = parse_IRC_xyz(IRC_xyz,Natoms=len(target_E))

            # match reactant node
            if return_inchikey(E,G1,adj_mat_1,separate=True) == return_inchikey(target_E,target_RG,target_Radj,separate=True):
                RG,PG,Radj_mat,Padj_mat = G1,G2,adj_mat_1,adj_mat_2
            elif return_inchikey(E,G2,adj_mat_2,separate=True) == return_inchikey(target_E,target_RG,target_Radj,separate=True):
                RG,PG,Radj_mat,Padj_mat = G2,G1,adj_mat_2,adj_mat_1
            else:
                continue

            # obtain smiles and inchikeys
            _,TS_G = xyz_parse(TS_xyz)
            rsmiles_map,rsmiles = return_atommaped_smi(E,RG,Radj_mat),return_smi(E,RG,Radj_mat).split('.')
            psmiles_map,psmiles = return_atommaped_smi(E,PG,Padj_mat),return_smi(E,PG,Padj_mat).split('.')
            rinchikey,pinchikey = return_inchi(E,RG,Radj_mat),return_inchi(E,PG,Padj_mat)

            # compute thermo 
            DG = (F_298 - sum([E_dict[inchi[:14]]['F'] for inchi in rinchikey]))*627.5095
            DE = (SPE - sum([E_dict[inchi[:14]]['SPE'] for inchi in rinchikey]))*627.5095
            TCIT_dict = TCIT_prediction(rsmiles+psmiles,TCIT_config)
            try:
                DH = (sum([TCIT_dict[smi]["Hf_298"] for smi in psmiles]) - sum([TCIT_dict[smi]["Hf_298"] for smi in rsmiles]))/4.184
            except:
                DH = 'TCIT fails'

            if len(rinchikey) > 1: rinchikey = '-'.join([inchi[:14] for inchi in rinchikey])
            else: rinchikey = rinchikey[0]

            if len(pinchikey) > 1: pinchikey = '-'.join([inchi[:14] for inchi in pinchikey])
            else: pinchikey = pinchikey[0]

            reaction_inchi = '-'.join(sorted([rinchikey,pinchikey]))
            
            # determine whether reaction_inchi exists in the reaction dictionary
            if reaction_inchi not in reactions.keys():
                reactions[reaction_inchi] = {'1':[Rindex,rsmiles_map,psmiles_map,DE,DG,DH,E,RG,PG,TS_G]}
                reactions[reaction_inchi]['energy'] = {DG:'1'}
            else:
                index = check_dup(reactions[reaction_inchi]['energy'].keys(),DG,thresh=0.5)
                if not index:
                    nind = len(reactions[reaction_inchi])
                    reactions[reaction_inchi][str(nind)] = [Rindex,rsmiles_map,psmiles_map,DE,DG,DH,E,RG,PG,TS_G]
                    reactions[reaction_inchi]['energy'][DG]=str(nind)


    for reaction_inchi,reaction in reactions.items():
        for conf_ind,reaction_info in reaction.items():
            if conf_ind == 'energy': continue
            # unpack info
            Rindex,rsmiles_map,psmiles_map,DE,DG,DH,E,RG,PG,TS_G = reaction_info
            
            # write into csv 
            with open(output_csv, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([Rindex,rsmiles_map,psmiles_map,DE,DG,DH])
        
            # write reaction into a xyz file
            out_xyz = '{}/{}.xyz'.format(output_path,Rindex)
            xyz_write(out_xyz,E,RG)
            xyz_write(out_xyz,E,PG,append_opt=True)
            xyz_write(out_xyz,E,TS_G,append_opt=True,comment='DE: {}  DG: {} DH: {} (kcal/mol)'.format(DE, DG, DH))
            

    return

# Function to call TCIT to give thermochemistry predictions (in kJ/mol)
def TCIT_prediction(smiles_list,config_file):

    # load in configs
    configs = parse_configuration(config_file)

    # load in TCIT database
    if os.path.isfile(configs["tcit_db"]) is True:
        with open(configs["tcit_db"],"r") as f:
            TCIT_dict = json.load(f) 
    else:
        TCIT_dict = {}
        
    # check the compounds without TCIT predictions
    smiles_list = [smi for smi in smiles_list if smi not in TCIT_dict.keys()]

    # write smiles into config file
    with open(configs["target_file"],'w') as f:
        for M_smiles in smiles_list:
            f.write("{}\n".format(M_smiles))
            
    # run TCIT calculation
    output=subprocess.Popen("python {} -c {}".format(configs["tcit_path"],config_file),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')

    # re-load in TCIT database
    with open(configs["tcit_db"],"r") as f:
        TCIT_dict = json.load(f) 
        
    return TCIT_dict

# load in config
def parse_configuration(config_file):

    # Convert inputs to the proper data type
    if os.path.isfile(config_file) is False:
        print("ERROR in python_driver: the configuration file {} does not exist.".format(config_file))
        quit()

    # Process configuration file for keywords
    keywords = ["TCIT_path","input_type","target_file","database","G4_db","TCIT_db","ring_db","xyz_task"]
    keywords = [ _.lower() for _ in keywords ]

    list_delimiters = [ "," ]  # values containing any delimiters in this list will be split into lists based on the delimiter
    space_delimiters = [ "&" ] # values containing any delimiters in this list will be turned into strings with spaces replacing delimites
    configs = { i:None for i in keywords }    

    with open(config_file,'r') as f:
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
                                
                    # Break if keyword is encountered in a non-comment token without an argument
                    else:
                        print("ERROR in python_driver: enountered a keyword ({}) without an argument.".format(i))
                        quit()

    # Makesure detabase folder exits
    if os.path.isfile(configs["database"]) is False: 
        print("No such data base")
        quit()
            
    if os.path.isfile(configs["ring_db"]) is False:
        print("No such ring correction database, please check config file, existing...")
        quit()

    if os.path.isfile(configs["g4_db"]) is False:
        print("No such G4 result database, please check config file, existing...")
        quit()

    if len(os.listdir(configs["xyz_task"]) ) > 0:
        subprocess.Popen("rm {}/*".format(configs["xyz_task"]),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
   
    return configs

# Function to Return smiles string 
def return_inchi(E,G,adj_mat=None,namespace='obabel'):

    if adj_mat is None:
        adj_mat = Table_generator(E, G)

    # Seperate reactant(s)
    gs      = graph_seps(adj_mat)
    groups  = []
    loop_ind= []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group =[count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
            loop_ind += new_group
            groups   +=[new_group]
            
    # Determine the inchikey of all components in the reactant
    inchi_list = []
    for group in groups:
        N_atom = len(group)
        frag_E = [E[ind] for ind in group]
        frag_adj = adj_mat[group][:,group]
        frag_G = np.zeros([N_atom,3])
        for count_i,i in enumerate(group): frag_G[count_i,:] = G[i,:]

        # generate inchikeu
        mol_write("{}_input.mol".format(namespace),frag_E,frag_G,frag_adj)
        substring = "obabel -imol {}_input.mol -oinchikey".format(namespace)
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        inchi  = output.split()[0]
        inchi_list += [inchi]
        
    os.system("rm {}_input.mol".format(namespace))

    return inchi_list

# function to identify if the target number matches with (closes to) a list of numbers
def check_dup(numbers,check_number,thresh=0.5):
    diff = [abs(check_number-number) for number in numbers]
    if min(diff) < thresh:
        return True
    else:
        return False

if __name__ == "__main__":
    main()

