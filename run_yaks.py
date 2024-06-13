import sys,os,argparse,fnmatch,subprocess,time,shutil,random, openpyxl
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import pickle
from write_yaml import smiles_to_inchi
from write_yaml import write_yaml_file
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import Descriptors

# all functions in taffi
sys.path.append(os.getcwd()+'/utilities')
from taffi_functions import *
from xtb_functions import xtb_energy,xtb_geo_opt
from job_submit import *
from utility import *
# from utility import return_smi,return_inchikey,parse_smiles

# Author: Michael Woulfe
""" Along with yaks_config.txt (same config file you use for write_yaml.py), runs cantera microkinetic modeling simulation and prints results for future exploration

Notes to self: make a deliberate restart scheme to take a number of folders as an input
"""
def main(argv):

    parser = argparse.ArgumentParser(description='Driver script for submitting YARP jobs with Gaussian engine.')

    #optional arguments                                             
    parser.add_argument('-c', dest='config', default='yaks_config.txt',
                        help = 'The program expects a configuration file from which to assign various run conditions. (default: yaml_config.txt in the current working directory).  Conveniently, write_yaml.py and run_yaks.py run off same yaml file.')
    
    print("parsing configuration directory...")
    args=parser.parse_args()
    config_path =args.config
    c = parse_configuration(parser.parse_args())
    run_yaks(c, config_path)

    return


def run_yaks(c,config_path):

    # get useful paths as pointers
    yaks_path = os.getcwd()
    yaks_config_path = f"{yaks_path}/{config_path}"

    # load in reactant/smi2inchi dictionaries
    reactant_dict = c["reactant_dict"]
    # print(reactant_dict)
    smi2inchi = c["smi2inchi"]
    print("Loading Smile to Inchi Dict...")
    if os.path.isfile(smi2inchi):
        with open(smi2inchi,'rb') as f: 
            smi_inchi_dict = pickle.load(f)
    else: smi_inchi_dict = {}

    # make folder to store configs/output files and YARP outputs
    if os.path.isdir(c["calc_folder"]) is False: os.mkdir(c["calc_folder"])
    if os.path.isdir(c["output_path"]) is False: os.mkdir(c["output_path"])

    
    node_tracker = {}
    previous_nodes = []
    bimol_combos = []

    # get reactant smile
    if len(c["initial_reactants"].split(','))==1:
        ini_smiles = c["initial_reactants"].split(':')[0]
    # or get reactant smiles
    else:
        ini_smiles = []
        # input: smile:0.5,smile2:0.3,smile3:0.2
        for i in c["initial_reactants"].split(','):
            ini_smiles.append(i.split(':')[0]) 
            # set bimole flag
        bimol_combos = list(combinations(ini_smiles, 2))

    # create folder to store bimolecular xyz's
    if os.path.isdir(f"{c['calc_folder']}/bimol_xyz") is False: os.mkdir(f"{c['calc_folder']}/bimol_xyz")
    
    # add initial nodes to calc list
    if len(previous_nodes)== 0 and type(ini_smiles) == str:
        new_nodes = [ini_smiles]
    else:
        new_nodes = ini_smiles

    smi_inchi_dict, _ = smiles_to_inchi(new_nodes, smi_inchi_dict) # create lists of inchis/ update smi_inchi_dict
    for i in bimol_combos:
        smi_inchi_dict, _ = smiles_to_inchi([i[0], i[1]], smi_inchi_dict) # create lists of inchis/ update smi_inchi_dict


    exp_steps = int(c['exp_steps'])
    ############## start yaks loop ##############
    for depth in range(exp_steps):
        print("depth", depth)

        # update smile to inchi dict
        smi_inchi_dict, _ = smiles_to_inchi(new_nodes, smi_inchi_dict)

        # write out xyz files of bimolecular explorations         
        if bimol_combos:
            print("Writing bimol_combos\n")
            for i in bimol_combos:
                print(i)

                # add bi_mol reactants to smi_inchi_dict
                smi_inchi_dict, _ = smiles_to_inchi([i[0], i[1]], smi_inchi_dict) # create lists of inchis/ update smi_inchi_dict

                # parse each combination of Elements and Geometries from smiles
                readin,E,G,q = parse_smiles(i[0],ff='mmff94',steps=100)
                readin2,E2,G2,q2 = parse_smiles(i[1],ff='mmff94',steps=100)

                # if both sucesfully parsed, write them into .xyz files
                if readin and readin2: 
                    print("both read in")
                    input1 = f"{c['calc_folder']}/bimol_xyz/{i[0]}_input.xyz"
                    input2 = f"{c['calc_folder']}/bimol_xyz/{i[1]}_input.xyz"
                    xyz_write(input1,E,G, q_tot=q)
                    xyz_write(input2,E2,G2, q_tot=q2)

                    # write a single optimized xyz file
                    combine_xyz_files(input1,input2,f"{c['calc_folder']}/bimol_xyz/{i[0]}-{i[1]}_output.xyz") 
                                                        # final file name will be opt_{i[0]}-{i[1]}_output.xyz
        else: print("no bimol combos at this step")


        # Reaction Enumeration
        print("Running ERS_enumeration...")
        inchi_list, bi_inchi_list = run_ERS(c, new_nodes, depth, yaks_path, bimol_combos)

        # record inchis in node tracker
        for num, inchi in enumerate(inchi_list):
            print(num, inchi)
            # node tracker[k] = [inchi, smile]
            node_tracker[f"{depth}_{num}"] = [inchi, new_nodes[num]]

        # add bimolecular info to node tracker
        for num, inchi in enumerate(bi_inchi_list):
            # find_key() reverse looks up key (smiles) based on value (inchi)
            node_tracker[f"{depth}_bi_{num}"] = [inchi, f"{find_key(smi_inchi_dict, inchi.split('-')[0])} + {find_key(smi_inchi_dict, inchi.split('-')[1])}"]

        print(node_tracker)

        # Run YARP exploration
        print("Running YARP exploration...")
        run_yarp_exp(c, node_tracker,depth, yaks_config_path)

        # if water emerges as major product, begin water-catalyzing reactions
        if 'XLYOFNOQVPJJNP' in node_tracker:
            c["water_cat"] = True # need to go back and water catalyze previous rxns too

        if c["water_cat"] == True:
            print("Running water catalyzed YARP exploration...")
            run_water_cat(c, node_tracker, depth, yaks_path, yaks_config_path)

        # # write yaml file
        print("Writing .yaml file...")
        file_name = c['calc_folder'].split('/')[-1]

        input_folds = set()
        # give current and preceding depth input folders to write_yaml
        for i in range(int(depth)+1):
            for k, v in node_tracker.items():

                # track current depth and all previous depths
                if fnmatch.fnmatch(k, f"{i}_*"):
                    input_folds.add(k)

                    # if water catalyzed, add those explorations too
                    if c["water_cat"] == True:
                        input_folds.add(f"{k}_wc")

        # set to list
        input_folds = list(input_folds)


        # write yaml input file
        write_yaml_file(c, folder1 = c['output_path'], folder2 = 'NA', file_name = file_name, input_folds = input_folds)
        print("input_folds",input_folds)

        # add depth to write_yaml's typical output
        if os.path.exists(f"{c['output_path']}/{file_name}.yaml"):
            os.rename(f"{c['output_path']}/{file_name}.yaml", f"{c['output_path']}/{file_name}_{depth}.yaml")
        if os.path.exists(f"{c['output_path']}/{file_name}_{depth}.yaml"):
            print(f"{file_name}_{depth}.yaml succesfully created")
        else: 
            print(f"Error: {file_name}_{depth}.yaml was not created")
            return
        
        # print("pre run_kinetics")
        # print('previous_nodes', previous_nodes)
        # print('new_nodes', new_nodes)
        # print('bimol_combos', bimol_combos)

        # add explored nodes to list of previous explorations
        previous_nodes += new_nodes 
        # append bimol exps to previous nodes list
        for i in bimol_combos:
            previous_nodes.append(f"{i[0]} + {i[1]}")
        
        # run kinetic simulation
        previous_nodes, new_nodes, bimol_combos = run_kinetics(c, f"{file_name}_{depth}.yaml", previous_nodes, depth)

        # print("post run_kinetics")
        print('previous_nodes', previous_nodes)
        print('new_nodes', new_nodes)
        print('bimol_combos', bimol_combos)
        print(node_tracker)

###########################################
####              FUNCTIONS            ####
###########################################

# Description: function that writes water catalysis submit file and submits it, performs conformational sampling on output, then conducts a type 2 yarp calc
#              on conformational sampling outputs 
#
# Inputs      c                 : yaks args
#             node_tracker      : dictionary of k:v --> {depth}_{species_number}: [inchi, smiles]
#             depth             : present exploration depth
#             yaks_path         : current working directory of yaks folder
#             yaks_config_path  : path to yaks_config.txt 
#
# Returns     Nothing.  Submits and monitors catalysis, conformational sampling, and yarp jobs
#
def run_water_cat(c, node_tracker, depth, yaks_path, yaks_config_path):
    
    work_dir = c['calc_folder']
    # node tracker[k] = [inchi, smile]

    # k = f"{depth}_{num}"
    wc_jobs = []
    for k, v in node_tracker.items():
        # identify current depth's files
        if fnmatch.fnmatch(k, f"{depth}_*"):

            # make directory to store water catalyzed calcs
            if os.path.isdir(f"{work_dir}/yarp_{k}/water_cat") is False: os.mkdir(f"{work_dir}/yarp_{k}/water_cat")
            if os.path.isdir(f"{work_dir}/yarp_{k}/water_cat/test-in") is False: os.mkdir(f"{work_dir}/yarp_{k}/water_cat/test-in")
            if os.path.isdir(f"{work_dir}/yarp_{k}/water_cat/test-out") is False: os.mkdir(f"{work_dir}/yarp_{k}/water_cat/test-out")

            input_files_conf_path = f"{c['output_path']}/{k}/input_files_conf"
            test_in_dir = f"{work_dir}/yarp_{k}/water_cat/test-in"

            # skip if you've already completed on a previous cycle/attampt
            if len(os.listdir(f"{work_dir}/yarp_{k}/water_cat/test-out")) > 1:
                print(f"test out directory full, skip to conf_gen for {k}")
                continue

            # Avoid rerunning water catalyzed yarp for small molecules without many conformations
            # conditions are: Few conformations, small smiles string of fewer than 7 characters, and previously copied values in test-in/
            elif len(os.listdir(input_files_conf_path)) < 10 and len(v[1]) < 7 and len(os.listdir(test_in_dir)) > 1: 
               print(f"small molecule, skip to conf_gen for {k}")
               continue

            # copy input_files_conf files to correct test-in directory
            if os.path.isdir(input_files_conf_path) is True:
                # check if files are in input_files_conf directory
                if any(os.path.isfile(os.path.join(input_files_conf_path, item)) for item in os.listdir(input_files_conf_path)):

                    # make a list of files in input_files_conf
                    files = [f for f in os.listdir(input_files_conf_path) if os.path.isfile(os.path.join(input_files_conf_path, f))]

                    # copy files to test-in directory
                    for file in files:
                        source_path = os.path.join(input_files_conf_path, file)
                        destination_path = os.path.join(test_in_dir, file)
                        shutil.copy(source_path, destination_path)

                # write generate_catalysis.py jobs 
                write_generate_cat_submit(c, f"{work_dir}/yarp_{k}/water_cat/wc_{k}.submit", k, yaks_path)
                
                # maintain list of files to submit
                wc_jobs.append(f"{work_dir}/yarp_{k}/water_cat/wc_{k}.submit")
                # print(wc_jobs)

    # submit jobs
    print("submitting catalysis jobs")
    wc_monitor = []
    for job in wc_jobs:
        command = f"sbatch {job}" # submit wc jobs
        output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        print(output)
        for i in output.split(): wc_monitor.append(i)

    # monitor generate catalysis jobs
    monitor_jobs(wc_monitor)
    
    conf_jobs = []
    # write and submit CONF_GEN jobs
    for k, v in node_tracker.items():

        # identify current depth
        if fnmatch.fnmatch(k, f"{depth}_*"):
            
            # if len(os.listdir(f"{c['output_path']}/inp_{k}_wc/input_files_conf")) <15:
            #     print(os.path.isdir(f"{c['output_path']}/inp_{k}_wc/input_files_conf"))
            #     print(len(os.listdir(f"{c['output_path']}/inp_{k}_wc/input_files_conf")))

            # skip conf gen calcs if you've already completed on a previous cycle/attampt
            if os.path.isdir(f"{c['output_path']}/inp_{k}_wc/input_files_conf") is True:

                # if the directory has .xyz files inside of it
                if len(os.listdir(f"{c['output_path']}/inp_{k}_wc/input_files_conf")) > 0:
                    print(f"inp_{k}_wc/input_files_conf directory full, skip to wc_yarp for {k}")
                    continue

                # same condidtion as water cat skip for small molecules
                # if we skip water catalysis rerun, we skip conformational sampling
                elif len(os.listdir(f"{c['output_path']}/{k}/input_files_conf")) < 10 and len(v[1]) < 7 and len(os.listdir(test_in_dir)) > 1: 
                    print(f"small molecule, skip conf_gen for {k}")
                    continue

            # copy conf_gen.py over (keeps submit and error files from crowding yaks folder)
            shutil.copy2(f"{'/'.join(work_dir.split('/')[:-1])}/yaks/conf_gen.py", f"{work_dir}/yarp_{k}/water_cat/conf_gen.py")
            shutil.copy2(f"{'/'.join(work_dir.split('/')[:-1])}/yaks/analyze_crest_output.py", f"{work_dir}/yarp_{k}/water_cat/analyze_crest_output.py")

            # write submit files
            write_conf_gen_submit(c, f"{work_dir}/yarp_{k}/water_cat/conf_gen_{k}.submit", k)
            conf_jobs.append(f"{work_dir}/yarp_{k}/water_cat/conf_gen_{k}.submit")

    # submit  and monitor conf_gen jobs
    print("submitting conf gen jobs")
    conf_monitor = []
    for job in conf_jobs:
        command = f"sbatch {job}" 
        output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        print(output)
        for i in output.split(): conf_monitor.append(i)
    monitor_jobs(conf_monitor)

    # write and submit yarp jobs
    for k, v in node_tracker.items():

        # identify current depth
        if fnmatch.fnmatch(k, f"{depth}_*"):
            
            # if conf_gen output folder exists
            if os.path.isdir(f"{c['output_path']}/inp_{k}_wc/input_files_conf"):

                # make working directory folder to hold output files and error messages
                if os.path.isdir(f"{work_dir}/yarp_{k}_wc") is False: os.mkdir(f"{work_dir}/yarp_{k}_wc")

                # write config file
                write_yarp_config(c,c['config_template'], yaks_config_path, f"{work_dir}/yarp_{k}_wc/{k}_config.txt", f"{c['output_path']}/{k}_wc", k, v, 2, input_xyz = f"{c['output_path']}/inp_{k}_wc/input_files_conf") 
                
                # write yarp_submit
                write_yarp_submit(c, f"{work_dir}/{k}_wc.submit", k, wc = 'yes')

                # copy critical scripts into the correct calc directory
                shutil.copy2(f"{'/'.join(work_dir.split('/')[:-1])}/yaks/locate_TS.py", f"{work_dir}/yarp_{k}_wc/locate_TS.py")
                shutil.copy2(f"{'/'.join(work_dir.split('/')[:-1])}/yaks/analyze_crest_output.py", f"{work_dir}/yarp_{k}_wc/analyze_crest_output.py")
                shutil.copy2(f"{'/'.join(work_dir.split('/')[:-1])}/yaks/refine.py", f"{work_dir}/yarp_{k}_wc/refine.py")

                # make directories necessary to run yarp
                if os.path.isdir(f'{c["calc_folder"]}/yarp_{k}_wc/xyz_files') is False: os.mkdir(f'{c["calc_folder"]}/yarp_{k}_wc/xyz_files')
                if os.path.isdir(f'{c["calc_folder"]}/yarp_{k}_wc/DFT') is False: os.mkdir(f'{c["calc_folder"]}/yarp_{k}_wc/DFT')

    # submit water catalyzed yarp jobs
    print("submitting wc yarp jobs")
    wc_yarp_jobs = []
    for file in os.listdir(work_dir):

        # only submit wc jobs, not normal yarp calcs
        if fnmatch.fnmatch(file, f"{depth}_*_wc.submit"):

            # first check if the yarp calc has already been completed
            # paths to report files
            irc_path = f"{c['output_path']}/{file.split('.')[0]}/IRC-result/IRC-record.txt"
            report_path = f"{c['output_path']}/{file.split('.')[0]}/report.txt"

            # base completion off report.txt or IRC-record.txt
            if c["low-irc"] == True:
                if os.path.exists(report_path) and os.path.getsize(report_path) > 200:
                    print("skipping", file, "report already completed")
                    continue # don't resubmit 
            elif c["low-irc"] == False:
                if os.path.exists(irc_path) and os.path.getsize(irc_path) > 200: # anything less than 196 is just an empty IRC-record.txt
                    print("skipping", file, "IRC-record already completed")
                    continue # don't resubmit 
            
            # same condidtion as water cat skip for small molecules (only applies if they've already been run once)
            # if we skip water catalysis/ conformational sampling, we skip wc_yarp on empty outputs
            
            elif len(os.listdir(f"{c['output_path']}/{k}/input_files_conf")) < 10 and len(v[1]) < 7 and len(os.listdir(test_in_dir)) > 1:
                print(f"small molecule, skip redoing wc_yarp for {k}")
                continue

            # submit unfinished yarp jobs
            command = f"sbatch {work_dir}/{file}" 
            output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            print(output)
            for i in output.split(): wc_yarp_jobs.append(i)
    monitor_jobs(wc_yarp_jobs)

    # save scratch file space
    # remove inp_*_*_wc jobs after finishing wc yarp calcs
    # for k, v in node_tracker.items():
    #     # identify current depth's files
    #     if fnmatch.fnmatch(k, f"{depth}_*"):
    #         os.system(f'rm {c["calc_folder"]}/inp_{k}_wc')

    # save scratch file space by cleaning folders after use?

# Description: writes submit file to run conformational sampling on water catalysis jobs
#
# Inputs      c                 : yaks args
#             submit_file       : file path and name
#             k                 : key of node_tracker
#
# Returns     Writes submit file, returns nothing.
#            
def write_conf_gen_submit(c, submit_file, k):
    sched = c['sched']
    work_dir = c['calc_folder']
    if sched == "slurm":    
        with open(f"{submit_file}",'w') as f:    

            # write header
            f.write("#!/bin/bash\n")
            f.write("#\n")
            f.write(f"#SBATCH --job-name={submit_file.split('/')[-1]}\n")
            f.write(f"#SBATCH --output={'.'.join(submit_file.split('.')[:-2])}.out\n")
            f.write(f"#SBATCH --error={'.'.join(submit_file.split('.')[:-2])}.err\n")
            f.write(f"#SBATCH -A bsavoie\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --ntasks-per-node=2\n") # more nodes for memory requirement
            f.write(f"#SBATCH --time 12:00:00\n")  

            # print out information
            f.write("\n# cd into the submission directory\n")
            f.write(f"cd {work_dir}/yarp_{k}/water_cat\n")
            f.write(f"echo Working directory is ${work_dir}/yarp_{k}/water_cat\n")
            f.write("echo Running on host `hostname`\n")
            f.write("echo Start Time is `date`\n")
            f.write('export PATH="/depot/bsavoie/apps/anaconda3/bin/:$PATH"\n')
            f.write("source activate python3\n")

            # run conformational sampling script
            f.write(f"python conf_gen.py {work_dir}/yarp_{k}/water_cat/test-out -N {c['n_max']} -Njob 8 -w {c['output_path']}/inp_{k}_wc -ff {c['ff']} -s 2 --remove_constraints\n")

            f.write("\nwait\n")
            f.write("echo End Time is 'date'\n")

# Description: writes submit file to run water catalysis jobs 
# 
# Inputs      c                 : yaks args
#             submit_file       : file path and name
#             k                 : key of node_tracker
#             yaks_path         : path to yaks directory
#
# Returns     Writes submit file, returns nothing.
#     
def write_generate_cat_submit(c, submit_file, k, yaks_path):
    sched = c['sched']
    work_dir = c['calc_folder']
    if sched == "slurm":    
        with open(f"{submit_file}",'w') as f:  
            
            # write header
            f.write("#!/bin/bash\n")
            f.write("#\n")
            f.write(f"#SBATCH --job-name={submit_file.split('/')[-1]}\n")
            f.write(f"#SBATCH --output={'.'.join(submit_file.split('.')[:-1])}.out\n")
            f.write(f"#SBATCH --error={'.'.join(submit_file.split('.')[:-1])}.err\n")
            f.write(f"#SBATCH -A standby\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --cpus-per-task=1\n") # more nodes for memory requirement
            f.write(f"#SBATCH --time 4:00:00\n") # ten days 

            # print out information
            f.write("\n# cd into the submission directory\n")
            f.write(f"cd {yaks_path}\n")
            f.write(f"echo Working directory is ${yaks_path}\n")
            f.write("echo Running on host `hostname`\n")
            f.write("echo Start Time is `date`\n")
            f.write('export PATH="/depot/bsavoie/apps/anaconda3/bin/:$PATH"\n')
            f.write("source activate python3\n")

            # write catalysis command
            f.write(f"python {yaks_path}/generate_catalysis.py -i {work_dir}/yarp_{k}/water_cat/test-in -o {work_dir}/yarp_{k}/water_cat/test-out -ff {c['ff']} --exclude_b3f3\n")
            f.write("\nwait\n")
            f.write("echo End Time is 'date'\n")

# Description: writes yarp config file, submission file, checks if jobs have already been completed, and then submits remaining jobs
# 
# Inputs      c                 : yaks args
#             node_tracker      : dictionary of k:v --> {depth}_{species_number}: [inchi, smiles]
#             depth             : present exploration depth
#             yaks_path         : current working directory of yaks folder
#
# Returns     Returns nothing, submits yarp jobs
#     
def run_yarp_exp(c, node_tracker, depth, yaks_path):
    
    work_dir = c['calc_folder']

    for k, v in node_tracker.items():
        if fnmatch.fnmatch(k, f"{depth}_*"):

            # make working directory folders
            if os.path.isdir(f"{work_dir}/yarp_{k}") is False: os.mkdir(f"{work_dir}/yarp_{k}")
            
            # write input_list_{depth}_{node}.txt
            with open(f"{work_dir}/yarp_{k}/input_list_{k}.txt",'w') as g:
                if isinstance(v[0], str):
                    for i in v[0]:                                                                 
                        g.write(f"{i}")

            # write yarp config file
            write_yarp_config(c,c['config_template'], yaks_path, f"{work_dir}/yarp_{k}/{k}_config.txt", f"{c['output_path']}/{k}", k, v, 0, input_xyz = f"{c['output_path']}/{k}/input_files_conf") 
            
            # write yarp_submit
            write_yarp_submit(c, f"{work_dir}/{k}.submit", k, wc = 'no')

            # copy utilities folder to working dir (forces any updates to copy over)
            shutil.copytree(f"{'/'.join(work_dir.split('/')[:-1])}/yaks/utilities",f"{work_dir}/utilities", dirs_exist_ok=True)

            # copy critical scripts into the correct calc directory
            shutil.copy2(f"{'/'.join(work_dir.split('/')[:-1])}/yaks/locate_TS.py", f"{work_dir}/yarp_{k}/locate_TS.py")
            shutil.copy2(f"{'/'.join(work_dir.split('/')[:-1])}/yaks/analyze_crest_output.py", f"{work_dir}/yarp_{k}/analyze_crest_output.py")
            shutil.copy2(f"{'/'.join(work_dir.split('/')[:-1])}/yaks/refine.py", f"{work_dir}/yarp_{k}/refine.py")

            # make directories necessary to run yarp
            if os.path.isdir(f'{c["calc_folder"]}/yarp_{k}/xyz_files') is False: os.mkdir(f'{c["calc_folder"]}/yarp_{k}/xyz_files')
            if os.path.isdir(f'{c["calc_folder"]}/yarp_{k}/DFT') is False: os.mkdir(f'{c["calc_folder"]}/yarp_{k}/DFT')

    # submit yarp jobs to cluster 
    all_yarp_jobs = []
    for file in os.listdir(work_dir):
        if fnmatch.fnmatch(file, f"{depth}_*.submit") and "_wc" not in file: # yarp submit files, but not wc

            # first check if the yarp calc has already been completed            
    
            # paths to report files
            yarp_out_fold = f"{c['output_path']}/{file.split('.')[0]}"
            irc_path = f"{yarp_out_fold}/IRC-result/IRC-record.txt"
            report_path = f"{yarp_out_fold}/report.txt"

            # base completion off report.txt or IRC-record.txt
            if c["low-irc"] == True:
                if os.path.exists(report_path) and os.path.getsize(report_path) > 200:
                    print("skipping", file, "report already completed")
                    continue # don't resubmit 
            elif c["low-irc"] == False:
                if os.path.exists(irc_path) and os.path.getsize(irc_path) > 200: # anything less is just an empty IRC-record.txt
                    print("skipping", file, "IRC-record already completed")
                    continue # don't resubmit 
            
            # manual exception to ensure we don't run yarp calc for water repeatedly
            if os.path.exists(f"{yarp_out_fold}/opt-folder/") and os.path.isdir(f"{yarp_out_fold}/opt-folder/"):
                if "XLYOFNOQVPJJNP.xyz" in os.listdir(f"{yarp_out_fold}/opt-folder/"):
                    print("skipping", file, "water has no unimolecular reactions")
                    continue # don't resubmit

            # submit unfinished yarp jobs
            command = f"sbatch {work_dir}/{file}" 
            output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            print(output)
            for i in output.split(): all_yarp_jobs.append(i)
            # all_yarp_jobs.append(output.split())
    monitor_jobs(all_yarp_jobs)

# Description: writes yarp submit file
# 
# Inputs      c                 : yaks args
#             submit_file       : file path and name
#             k                 : key of node_tracker
#             wc                : 'no' or 'yes', determines whether the submit file is for wc or normal yarp calcs 
#
# Returns     Writes submit file, returns nothing.
#     
def write_yarp_submit(c, submit_file, k, wc = ''):
    # for water catalyzed
    if wc is None:
        wc = 'no'
    elif wc.lower() == 'no': wc = 'no'
    elif wc.lower() == 'yes': wc = 'yes'
    else:
        print("Error: write_yarp_submit.  Must write 'no' or 'yes' for water-catalyzed, 'wc' ")
        quit()

    # for normal
    sched = c['sched']
    work_dir = c['calc_folder']
    if sched == "slurm":    
        with open(f"{submit_file}",'w') as f: 

            # header for submit file
            f.write("#!/bin/bash\n")
            f.write("#\n")
            if wc == 'no': f.write(f"#SBATCH --job-name=yarp_{submit_file.split('/')[-1]}\n")
            elif wc == 'yes': f.write(f"#SBATCH --job-name=yarp_{submit_file.split('/')[-1]}_wc\n")
            f.write(f"#SBATCH --output={'.'.join(submit_file.split('.')[:-1])}.out\n")
            f.write(f"#SBATCH --error={'.'.join(submit_file.split('.')[:-1])}.err\n")
            f.write(f"#SBATCH -A bsavoie\n") # account to run on cluster 
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --cpus-per-task=4\n") # more nodes for memory requirement
            f.write("#SBATCH --mem-per-cpu=1G\n")  # for large reaction dictionaries (glucose network with 500+ explorations)
            f.write(f"#SBATCH --time 72:00:00\n") # three days (should be enough time for most molecules)


            # print out information
            f.write("\n# cd into the submission directory\n")
            if wc == 'no': f.write(f"cd {work_dir}/yarp_{k}\n")
            elif wc == 'yes': f.write(f"cd {work_dir}/yarp_{k}_wc\n")
            if wc == 'no': f.write(f"echo Working directory is ${work_dir}/yarp_{k}\n")
            elif wc == 'yes': f.write(f"echo Working directory is ${work_dir}/yarp_{k}_wc\n")

            # ensure proper environment
            f.write("echo Running on host `hostname`\n")
            f.write("echo Start Time is `date`\n")
            f.write('export PATH="/depot/bsavoie/apps/anaconda3/bin/:$PATH"\n')
            f.write("source activate python3\n")


            # run yarp
            f.write(f"python locate_TS.py -c {k}_config.txt\n")
            f.write("\nwait\n")
            f.write("echo End Time is 'date'\n")

# Description: writes a config file for yarp based off a template and replaces key args with molecule specific values
# 
# Inputs      c                     : yaks args
#             config_template_path  : path to config_template
#             yaks_config_path      : path to yaks_config.txt
#             output_path           : path to yarp output directory
#             output_folder         : file path and name
#             k                     : key of node_tracker
#             v                     : [inchi, smiles] - values of node_tracker
#             inp_type              : 0 or 2 based on whether yarp is normal or water catalyzed
#             input_xyz             : file path to the input xyz files. If none entered, default is output dirrectory/input_files_conf
#
# Returns     Writes config file, returns nothing.
#     
def write_yarp_config(c, config_template_path, yaks_config_path, output_path, output_folder, k, v, inp_type, input_xyz=''):  
    
    # Read content from the template config file, and from yaks_config file
    modified_lines = []
    with open(config_template_path, 'r') as f, open(yaks_config_path, 'r') as g:

        # parse dictionary of yaks_config values
        config_dict = {}
        for line in g:
            desired_line = line.split('#')[0].strip()
            if len(desired_line.split()) == 2: # parse only lines with two values
                config_dict[desired_line.split()[0]] = desired_line.split()[1]

        # parse config_templates values (to replace with yaks_config values)
        for line in f:
            modified_line = line.split('#')[0].strip()
            if len(modified_line.split()) ==2:
                temp = modified_line.split()

                # for molecule specific args, use rdkit to parse charge and unpaired electrons
                # charge
                if temp[0] == "charge": 
                    mol = Chem.MolFromSmiles(v[1])
                    if mol:
                        formal_charge = Chem.GetFormalCharge(mol)
                        temp[1] = str(formal_charge)

                # unpaired electrons
                elif temp[0] == "unpair":
                    mol = Chem.MolFromSmiles(v[1])
                    if mol:
                        num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
                        temp[1] = str(num_unpaired_electrons)

                # input_type (water catalyzed 2 vs standard 0)
                elif temp[0] == "input_type":
                    temp[1] = str(inp_type)

                # newer version of yarp requires accurate input_files_conf path whether type 0 0or 2
                elif temp[0] == "input_xyz":
                    if temp[1] and input_xyz: temp[1] = input_xyz
                    elif temp[1]: temp[1] = f"{output_path}/input_files_conf"
                    else:
                        modified_line = f"{temp[0]:<20}\t{output_path}/input_files_conf"
                        modified_lines.append(modified_line)
                        continue

                # specific to depth and species
                elif temp[0] == "input_react":
                    temp[1] = str(f"input_list_{k}.txt")

                # all yarp calcs go into same folder
                elif temp[0] == "output_path":
                   temp[1] = str(output_folder)

                # or replace template values with yaks_config values
                else:
                    try:
                        temp[1]= str(config_dict[temp[0]])
                    except KeyError:
                        pass
                # print(temp, type(temp[1]))
                
                # compile list of new config file lines
                modified_line =  f"{temp[0]:<20}\t{temp[1]}"
                modified_lines.append(modified_line)
        
        # add temperature to yarp config file
        modified_lines.append(f"temperature         \t{c['calc_temp']}")
    # write new config file
    with open(output_path, 'w') as of:
        for modified_line in modified_lines:
            of.write(f"{modified_line}\n")

# Description: get bounding box for an xyz file
# 
# Inputs      file                 : path to  xyz file
#             
# Returns     min and max coords of bounding box
#     
def get_bounding_box(file):
    with open(file, 'r') as f:
        next(f)  # Skip the first line
        coords = [list(map(float, line.split()[1:])) for line in f if len(line.strip()) > 0 and line.split()[0].isalpha()]
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        return min_coords, max_coords
    
# Description: optimizes a given xyz file using openbabel
# 
# Inputs      output_file                 : path to  xyz file
#             forcefield                  : forcefield to use for optimization
#             
# Returns     min and max coords of bounding box
#     
def optimize_xyz_file(output_file, forcefield='MMFF94'):
    # Construct the optimized file path
    optimized_file_path = '/'.join(output_file.split('/')[:-1]) + f"/opt_{output_file.split('/')[-1]}"
    # Construct the command string
    command = f"obabel '{output_file}' -O '{optimized_file_path}' --sd --minimize --steps 1000 --ff {forcefield}"
    
    # Execute the command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decode and print output for debugging (optional)
    # print("STDOUT:", stdout.decode('utf-8'))
    # print("STDERR:", stderr.decode('utf-8'))
    
    return stderr.decode('utf-8')  # Return the exit code of the subprocess
# Description: read in two xyz files into a single xyz file and optimize it
# 
# Inputs      file1                 : path to first xyz file
#             file2                 : path to second xyz file
#             output_file           : path to output xyz file (will be named opt_{output_file})
#             offset_factor         : factor to offset the second molecule outside it's bounding box
# 
# Returns     Writes combined xyz file
#     

def combine_xyz_files(file1, file2, output_file, offset_factor=1.2):

    # Calculate the bounding box of each molecule
    min_coords1, max_coords1 = get_bounding_box(file1)
    min_coords2, max_coords2 = get_bounding_box(file2)

    # Calculate the size of each molecule
    size1 = max_coords1 - min_coords1
    size2 = max_coords2 - min_coords2

    # Calculate the offset based on the size of the molecules
    offset = max(np.max(size1), np.max(size2)) * offset_factor

    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as output:
        # Read the first line of each file to get the total number of atoms
        total_atoms_file1 = int(f1.readline().strip())
        total_atoms_file2 = int(f2.readline().strip())

        # Calculate the total number of atoms in the combined system
        total_atoms_combined = total_atoms_file1 + total_atoms_file2
        
        # Write the total number of atoms in the combined system
        output.write(str(total_atoms_combined) + '\n\n')

        # Copy coordinates from the first file
        for i in range(total_atoms_file1+1):
            line = f1.readline().strip()
            if len(line) > 0 and line.split()[0].isalpha():
                element, coords = line.split()[0], line.split()[1:]
                output.write(f"{element:<20s} {float(coords[0]):<20.8f} {float(coords[1]):<20.8f} {float(coords[2]):<20.8f}\n")

        # Copy coordinates from the second file with an offset in the x-direction
        for i in range(total_atoms_file2+1):
            line = f2.readline().strip()
            if len(line) > 0 and line.split()[0].isalpha():
                element, coords = line.split()[0], line.split()[1:]
                x, y, z = map(float, coords)  # Extract x, y, z coordinates
                new_x, new_y, new_z = x + offset, y + offset, z + offset
                output.write(f"{element:<20s} {new_x:<20.8f} {new_y:<20.8f} {new_z:<20.8f}\n")

    # Attempt optimization with MMFF94
    if optimize_xyz_file(output_file, 'MMFF94') == 'Could not setup force field.\n0 molecules converted\n':
        # If MMFF94 fails, attempt with UFF
        print("MMFF94 optimization failed, attempting with UFF...")
        optimize_xyz_file(output_file, 'UFF')
    # remove unoptimized geometry
    # os.remove(output_file)

# Description: Enumerate all reaction possibilities for new nodes.  writes ERS submit file for uni and bi reactions, submits file, 
#              and recovers lists of uni and bimoelcular inchis
# 
# Inputs      c                     : yaks args
#             new_nodes             : list of smiles strings to perform enumeration on
#             depth                 : exploration depth
#             yaks_path             : path to yaks directory
#             bimol_combos          : list of sets of smiles strings for two species that need to be bimolecularly combined
# 
# Returns     list of inchis for unimolecular reactions, list of inchis for bimolecular reactions
#     
def run_ERS(c, new_nodes, depth, yaks_path, bimol_combos): # depth is integer of exploration depth
    
    # gather values from config file
    work_dir = c['calc_folder']
    sched = c['sched']
    ers_queue = c['ers_queue']
    Wt = int(c['ers_walltime'])
    reactant_dict = c['reactant_dict']


    # determine walltime 
    if Wt >= 1: min_flag = True
    elif Wt < 1 and Wt > 0: 
        min_flag = False
        Wt = Wt*60

    # write submit file for unimolecular reaction
    if sched == "slurm":    
        with open(f"{work_dir}/ERS_{depth}.submit",'w') as f:

            # write out header information 
            f.write("#!/bin/bash\n")
            f.write("#\n")
            f.write(f"#SBATCH --job-name=ERS_{depth}\n")
            f.write(f"#SBATCH --output={work_dir}/ERS_{depth}.out\n")
            f.write(f"#SBATCH --error={work_dir}/ERS_{depth}.err\n")
            f.write(f"#SBATCH -A {ers_queue}\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --cpus-per-task=4\n")
            
            # assign wall-time
            if min_flag:
                f.write(f"#SBATCH --time {Wt}:00:00\n")
            else:
                f.write(f"#SBATCH --time 00:{Wt}:00\n")

            # print out information
            f.write("\n# cd into the submission directory\n")
            f.write(f"cd {yaks_path}\n")
            f.write(f"echo Working directory is ${yaks_path}\n")
            f.write("echo Running on host `hostname`\n")
            f.write("echo Start Time is `date`\n")
            
            # use correct environment
            f.write('export PATH="/depot/bsavoie/apps/anaconda3/bin/:$PATH"\n')
            f.write("source activate python3\n")

            # write ERS_enumeration unimolecular command
            if c['ers_type'] == 'pb3f3':
                f.write(f"python reaction_enumeration.py {work_dir}/inp_smiles_{depth}.txt -rd {reactant_dict} -ff {c['ff']} -P 1 --partial_b3f3\n")
            elif c['ers_type'] == 'b3f3':
                f.write(f"python reaction_enumeration.py {work_dir}/inp_smiles_{depth}.txt -rd {reactant_dict} -ff {c['ff']} -P 1 --b3f3\n")
            else: f.write(f"python reaction_enumeration.py {work_dir}/inp_smiles_{depth}.txt -rd {reactant_dict} -ff {c['ff']} -P 1 \n")

            # write ERS_enumeration bimolecular commands for every bimol combo
            if bimol_combos:
                if c['ers_type'] == 'pb3f3':
                    for i in bimol_combos:
                        # only run reaction enumeration if molecules succesfully written into .xyz file
                        if os.path.exists(f"{c['calc_folder']}/bimol_xyz/opt_{i[0]}-{i[1]}_output.xyz"):
                            f.write(f"python reaction_enumeration.py  '{c['calc_folder']}/bimol_xyz/opt_{i[0]}-{i[1]}_output.xyz' -rd {reactant_dict} -ff {c['ff']} -P 2 --partial_b3f3\n")
                if c['ers_type'] == 'b3f3':
                    for i in bimol_combos:
                        # only run reaction enumeration if molecules succesfully written into .xyz file
                        if os.path.exists(f"{c['calc_folder']}/bimol_xyz/opt_{i[0]}-{i[1]}_output.xyz"):
                            f.write(f"python reaction_enumeration.py  '{c['calc_folder']}/bimol_xyz/opt_{i[0]}-{i[1]}_output.xyz' -rd {reactant_dict} -ff {c['ff']} -P 2 --b3f3\n")
                else:
                    for i in bimol_combos:
                        # only run reaction enumeration if molecules succesfully written into .xyz file
                        if os.path.exists(f"{c['calc_folder']}/bimol_xyz/opt_{i[0]}-{i[1]}_output.xyz"):
                            f.write(f"python reaction_enumeration.py '{c['calc_folder']}/bimol_xyz/opt_{i[0]}-{i[1]}_output.xyz' -rd {reactant_dict} -ff {c['ff']} -P 2\n")

            f.write("\nwait\n")
            f.write("echo End Time is 'date'\n")

    # write new node smiles into inp_smiles.txt
    with open(f'{work_dir}/inp_smiles_{depth}.txt','w') as g:
        if isinstance(new_nodes, list):
            for i in new_nodes:                                                                 
                g.write(f"{i}\n")
        if isinstance(new_nodes, str):
            for i in new_nodes:                                                                 
                g.write(f"{i}")

    # submit jobs using CLI commands and .submit files
    command = f"sbatch {work_dir}/ERS_{depth}.submit"
    output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
    # wait until jobs are finished
    monitor_jobs(output.split())

    # retrieve inchis and bimolecular inchis
    inchi_list, bi_inchi_list = [], []
    with open(f"{work_dir}/ERS_{depth}.out", 'r') as file:
        for line in file:
            if "Inchi index is" in line:
                if len(line.split()[-1].split('-'))==2:
                    bi_inchi_list.append(line.split()[-1])
                else:
                    inchi_list.append(line.split()[-1])
    return inchi_list, bi_inchi_list

# Description: Runs kinetic simulation, returns list of previous nodes, new nodes to explore, and bimol_nodes to explore
# 
# Inputs      c                     : yaks args
#             yaml_name             : yaml_file name
#             previous_nodes        : list of previously explored species
# 
# Returns     list of inchis for unimolecular reactions, list of inchis for bimolecular reactions
#     
def run_kinetics(c, yaml_name, previous_nodes, depth):

    # get path to specific yaml file
    yaml = f"{c['output_path']}/{yaml_name}"
    time_step = float(c['time_step'])
    t_end = float(c['steps']) * time_step
    rule = c['rule']
    # sim.verbose = True # remove a few print outs

    # guide kinetics with Gaussian injected uncertainty
    if c['uncertainty_guided'] == True:
        conv = 4184000 # conv 1 kcal/mol = 4184000 j/kmol # default Cantera E units is J/kmol
        scale = float(c['uncertainty_range'])/2*conv  # 95% of errors will be within 3 kcal/mol when scale = 1.5 * conv
        tracker, conc, mass_p = [], [], []

        # begin kinetic simulation loop
        start_time = time.time()
        print("uncertainty guided start loop")
        for i in range(int(c['uncertainty_cycles'])):

            gas = ct.Solution(yaml)

            # initially determine gas mixture
            r = ct.IdealGasConstPressureReactor(contents=gas, energy='off', name='isothermal_reactor')
            sim = ct.ReactorNet([r])
            states = ct.SolutionArray(gas, extra=['t'])

            # get speies and reaction list
            species = states.species_names
            rxns = gas.reactions()

            # introduce noise
            noise = np.random.normal(0,scale, len(rxns))
            temp = {}
            sim_bar = []
            # update barriers
            for cou, val in enumerate(gas.reactions()):
                temp = rxns[cou].input_data
                temp['rate-constant']['Ea'] = (rxns[cou].rate.activation_energy + noise[cou])
                gas.modify_reaction(cou, ct.Reaction.from_dict(temp, kinetics = gas))
                # sim_bar.append(temp['rate-constant']['Ea']/conv) # only use if we want to track barrier heights

            # redefine reactor with noisy barriers
            r = ct.IdealGasConstPressureReactor(contents=gas, energy='off', name='isothermal_reactor')
            sim = ct.ReactorNet([r])
            states = ct.SolutionArray(gas, extra=['t'])
    
            ######## run simulation #############
            if depth == 0: states.append(r.thermo.state, t=sim.time)
            while sim.time <= t_end:
                sim.advance(sim.time + time_step)
                states.append(r.thermo.state, t=sim.time)

            ############ run analysis ############
            if rule == 'cnf':
                # parse depth from yaml name. If initial depth... 
                if yaml_name.split("_")[-1].split('.')[0] == 0:
                    # use every flux datapoint
                    spe_net = np.trapz(states.net_production_rates[:], dx=time_step, axis=0)
                # ignore first timestep
                else: spe_net = np.trapz(states.net_production_rates[1:], dx=time_step, axis=0)
                net_states = zip(spe_net,species,range(len(species)))
                net_states = sorted(net_states,reverse=True)[:]
                tracker.append(spe_net)

            elif rule == 'css':
                net_states = zip(states.X[-1,:],species,range(len(species)))
                net_states = sorted(net_states,reverse=True)[:]
                tracker.append(states.X[-1,:])
            
            elif rule == 'final_mass':
                net_states = zip(states.Y[-1,:],species,range(len(species)))
                net_states = sorted(net_states,reverse=True)[:]
                tracker.append(states.Y[-1,:])

            elif rule == 'cum_conc':
                conc_net = np.trapz(states.x[:], dx=time_step, axis=0)
                net_states = zip(conc_net,species,range(len(species)))
                net_states = sorted(net_states,reverse=True)[:]
                tracker.append(conc_net)
            
            # collect all mass percents, concentrations, and cnfs for each sim
            mass_p.append(states.Y[-1,:])
            conc.append(states.Y[-1,:])
        
        print(f"uncertainty guided loop end: {time.time()- start_time}")
        
        # conduct analysis
        results_list_of_dicts = calc_uncertainty_stats(tracker,species)

        # Sort the dictionaries for the highest n values of mean
        sorted_by_mean = sorted(results_list_of_dicts, key=lambda x: x['mean'], reverse=True)

        # Sort the dictionaries for the highest n values of median
        sorted_by_median = sorted(results_list_of_dicts, key=lambda x: x['median'], reverse=True)

        # rank states by the mean and put into noiseless formatting
        net_states = [(stat['mean'], stat['species'], stat['index']) for stat in sorted_by_mean]

        # write rates to excel spreadsheet, save in analysis folder
        with pd.ExcelWriter(f"{c['output_path']}/{yaml_name.split('.')[0]}_stats.xlsx") as writer:
            pd.DataFrame(tracker, columns=species).to_excel(writer, sheet_name=rule) # track rules for each sim
            pd.DataFrame(sim_bar, columns=rxns).to_excel(writer, sheet_name=rule) # track barrier heights
            pd.DataFrame(mass_p, columns=species).to_excel(writer, sheet_name='final_mass')    
            pd.DataFrame(conc, columns=species).to_excel(writer, sheet_name='final_concentration')
            pd.DataFrame(results_list_of_dicts).to_excel(writer, sheet_name='uncertainty_results')

    else:
        ############ Noiseless Analysis ###############
        print("Noiseless Sim")
        # define the reactor
        gas = ct.Solution(yaml) 
        if c['reactor'] == "IGCPR": r = ct.IdealGasConstPressureReactor(contents=gas, energy='off', name='isothermal_reactor')
        sim = ct.ReactorNet([r])
        states = ct.SolutionArray(gas, extra=['t'])

        # gives snapshot of system
        # print(gas(), f"T0: {gas.T}", f"Species: {len(gas.Y)}", f"Reactions: {len(states.reactions())}")

        species = states.species_names
        rxns = states.reactions()

        ######## run simulation #############
        if depth == 0: states.append(r.thermo.state, t=sim.time)
        while sim.time <= t_end:
            sim.advance(sim.time + time_step)
            states.append(r.thermo.state, t=sim.time)

        # depending on rule used to evaluate sims, order species greatest to least
        ############ run analysis ############
        if rule == 'cnf':
            if fnmatch.fnmatch(yaml, "*_0.yaml") or fnmatch.fnmatch(yaml, "*_pre*"): 
                spe_net = np.trapz(states.net_production_rates[:], dx=time_step, axis=0)
            else: spe_net = np.trapz(states.net_production_rates[1:], dx=time_step, axis=0)    
            net_states = zip(spe_net,species,range(len(species)))
            net_states = sorted(net_states,reverse=True)[:]

        elif rule == 'css':
            net_states = zip(states.X[-1,:],species,range(len(species)))
            net_states = sorted(net_states,reverse=True)[:]
        
        elif rule == 'final_mass':
            net_states = zip(states.Y[-1,:],species,range(len(species)))
            net_states = sorted(net_states,reverse=True)[:]

        elif rule == 'cum_conc':
            conc_net = np.trapz(states.x[:], dx=time_step, axis=0)
            net_states = zip(conc_net,species,range(len(species)))
            net_states = sorted(net_states,reverse=True)[:]

        # extra analysis
        # spe_cre = np.trapz(states.creation_rates, dx=time_step, axis=0)
        # spe_des = np.trapz(states.destruction_rates, dx=time_step, axis=0)
        # rxn_net = np.trapz(states.net_rates_of_progress, dx=time_step, axis=0)

        # write rates to excel spreadsheet, save in analysis folder
        with pd.ExcelWriter(f"{c['output_path']}/{yaml_name.split('.')[0]}_stats.xlsx") as writer:
            if rule == 'cnf': pd.DataFrame([spe_net], columns=species).to_excel(writer,sheet_name='cnf') 
            elif rule == 'cum_conc': pd.DataFrame([conc_net], columns=species).to_excel(writer,sheet_name='cum_conc') 
            pd.DataFrame(states.X, columns=species).to_excel(writer,sheet_name='concentration')    
            pd.DataFrame(states.Y, columns=species).to_excel(writer,sheet_name='mass_percent')
            pd.DataFrame(states.net_rates_of_progress, columns=rxns).to_excel(writer,sheet_name='rxn_fluxes')    
            pd.DataFrame(states.net_production_rates, columns=species).to_excel(writer,sheet_name='species_fluxes')        

    # make concentration/mass/net creation plots and save them to analysis folder (one folder for each step)

    # reset bimol combos, new nodes
    possible_bimole, bimol_combos, new_nodes = [], [], []
    high_value = net_states[0][0] # highest flux value (only used for cnf cutoff threshold below)

    print("previous:", previous_nodes)

    for cou, val in enumerate(net_states):

        # if cou <= int(c['num_nodes']):
        #     print(cou, val)

        # if we are using cnf and there is a large gap between the highest value and another one in the top 5,
        # skip low cnf values (save computational cost/avoid exploring down unfavorable pathways)
        # if rule == 'cnf' and c["thresh"] == 'yes':
        #     # print('cnf rule', c['thresh'], high_value,  i[0], val[0])
        #     try:
        #         # net states is ordered highest to lowest.  If diff between current species and top species > threshold, end analysis
        #         if float(high_value)/float(val[0])>= 10**float(c['cnf_thresh']):
        #             return previous_nodes, new_nodes, bimol_combos
        #     except ZeroDivisionError:   
        #         continue

        # check if top five cnf are 
        if cou < int(c['num_nodes']):
            # only conduct bimol calcs if species in previous exp step nodes
            if val[1] in previous_nodes: 
                possible_bimole.append(val[1])
        
        # update new nodes list
        if val[1] not in previous_nodes:
            new_nodes.append(val[1])

        # exit criteria: end for loop once we have correct number of nodes
        if len(new_nodes) == int(c['num_nodes']):
            break
    
    if c["bimolecular"] == True:
        # determine bimol combination for next round of exploration
        if len(possible_bimole) >= 2:
            bimol_combos = list(combinations(possible_bimole, 2))

        print("bimol_combos", bimol_combos)

        # check if one of the bimol combos is previously explored
        print("previous node vs bimol check")
        for node in previous_nodes:
            spt_lst = node.split(" + ")
            if fnmatch.fnmatch(node, '* + *'): # bimol node
                # print("bimol reactants")
                for i in bimol_combos:
                    # print("node", node)
                    # print("i's", i[0], i[1])
                    # if bimol combo consists of same reactant
                    if i[0] == i[1] and spt_lst[1] == spt_lst[0] and i[0] == spt_lst[0]:
                        bimol_combos.remove(i)               
                        # print("removed", i)

                    # or if both reactants are in previous nodes together, and they are different species
                    elif (i[0] == spt_lst[0] or i[0] == spt_lst[1]) and (i[1] == spt_lst[0] or i[1] == spt_lst[1]) and i[0] != i[1]:
                        bimol_combos.remove(i)
                        # print("removed", i)

                    # remove water as bimolecular reactantx 
                    # if val[0] == 'O' or val[1] == 'O':
                    #     bimol_combos.remove(i)
            
    return previous_nodes, new_nodes, bimol_combos

# Description: calculates uncertainty stats from mulitple noisy kinetic sim results
# 
# Inputs      list_of_lists         : list of lists of all kinetic simulation results (ordering kept constant)
#             species               : list of species
# 
# Returns     list of dictionaries of stats for each species

def calc_uncertainty_stats(list_of_lists, species):

    # Transpose the list of lists to group values by their index
    transposed = zip(*list_of_lists)
    
    # Initialize a list to hold statistics for each index
    stats = []
    
    # Calculate statistics for each group of values
    for original_index, (name, values) in enumerate(zip(species,transposed)):
        values = list(values)  # Convert tuple from zip to list for numpy functions
        mean = np.mean(values)
        median = np.median(values)
        maximum = np.max(values)
        minimum = np.min(values)
        quartile_25 = np.percentile(values, 25)
        quartile_75 = np.percentile(values, 75)
        num = len(values)
        
        # Append a dictionary of stats for each index
        stats.append({
            'species': name,
            'index': original_index,
            'mean': mean,
            'median': median,
            'max': maximum,
            'min': minimum,
            '25th percentile': quartile_25,
            '75th percentile': quartile_75,
            'number of values': num,
        })
    
    return stats

# Function for keeping tabs on the validity of the user supplied inputs
def parse_configuration(args):
    
    # Convert inputs to the proper data type
    if os.path.isfile(args.config) is False:
        print("ERROR in python_driver: the configuration file {} does not exist.".format(args.config))
        quit()
    
    # Process configuration file for keywords
    keywords = ["input_type","reactant_dict","reaction_dict","smi2inchi","output_path","pygsm_path","e_dict","ff","apply_tcit","tcit_result","hf_cut","charge","unpair","batch","sched","restart_step","eps",\
                "select_conf","low-irc","dft-irc","criterion","memory","irc-image","stepsize","irc-model","dg_thresh","compute_product","level","nimage","add-tor","conv-tor","relax_end","low_wt","low_njobs","low_queue","low_procs",\
                "c_method","c_wt","c_njobs","c_nprocs","n_max","c_queue","add-joint","c_path","functional","basis","dispersion","solvation","high_wt","ppn","high_procs","high_njobs","high_queue","parallel",\
                "rads","rev_rxns","energy_units","eos","kinetics","reactions","transport","temp","pres","pres_units","model_type","file_name","initial_reactants","EA_cutoff","ERS_queue","ERS_walltime",\
                "calc_folder", "config_template", "scratch_path","scratch_folds","depot_path","depot_folds","time_step","steps","rule","reactor","num_nodes","cnf_thresh","exp_steps","water_cat","input_react","input_xyz","thresh",\
                "uncertainty_guided", "uncertainty_cycles", "uncertainty_range", "bimolecular","calc_temp","ers_type"]

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
       
    # Check that batch is an acceptable system
    if configs["batch"] not in ["pbs","slurm"]:
        print("ERROR in locate_TS: only pbs and slurm are acceptable arguments for the batch variable.")
        quit()
    elif configs["batch"] == "pbs":
        configs["sub_cmd"] = "qsub"
    elif configs["batch"] == "slurm":
        configs["sub_cmd"] = "sbatch"

    # Check that dispersion option is valid
    if configs["dispersion"].lower() == 'none': 
        configs["dispersion"] = None
    elif configs["dispersion"] not in ["D2", "D3", "D3BJ"]:
        print("Gaussian only has D2, D3 and D3BJ Empirical Dispersion, check inputs...")
        quit()

    # Check that solvation option is valid
    if configs["solvation"].lower() == 'none': 
        configs["solvation"] = None
    elif '/' not in configs["solvation"]:
        print("expect A/B format where A refers to model and B refers to solvent")
        quit()

    # Check that relax end is an acceptable system
    if configs["restart_step"] not in ["None","GSM","low-TS","DFT-TS","low-IRC","IRC"]:
        print("ERROR in locate_TS: only None, GSM, low-TS, DFT-TS, low-IRC and IRC are acceptable")
        quit()

    # check settings
    if configs["criterion"] not in ["S","D"]: 
        print("Please specify a correct criterion: 'S' means if IRC nodes match reactant; 'D' means IRC node match either R/P")
        quit()

    # set default value
    # input_type (default:0)
    if configs["input_type"] is None:
        configs["input_type"] = 0
    else:
        configs["input_type"] = int(configs["input_type"])
    
    # apply TCIT
    if configs["apply_tcit"] is None:
        configs["apply_tcit"] = True

    elif configs["apply_tcit"].lower() == 'false':
        configs["apply_tcit"] = False

    else:
        configs["apply_tcit"] = True

    # fixed two end nodes
    if configs["relax_end"] is None:
        configs["relax_end"] = True
    elif configs["relax_end"].lower() == 'false':
        configs["relax_end"] = False
    else:
        configs["relax_end"] = True
    
    # apply joint optimization
    if configs["add-joint"] is None:
        configs["add-joint"] = True
    elif configs["add-joint"].lower() == 'false':
        configs["add-joint"] = False
    else:
        configs["add-joint"] = True

    # set default value for dg_thresh
    if configs["dg_thresh"] is None or configs["dg_thresh"].lower() == 'none':
        configs["dg_thresh"] = None
    else:
        configs["dg_thresh"] = int(configs["dg_thresh"])

    # apply product optimization
    if configs["compute_product"] is None:
        configs["compute_product"] = False
    elif configs["compute_product"].lower() == 'true':
        configs["compute_product"] = True
    else:
        configs["compute_product"] = False

    # remove duplicate TSs or not
    if configs["select_conf"] is None:
        configs["select_conf"] = False
    elif configs["select_conf"].lower() == 'false':
        configs["select_conf"] = False
    else:
        configs["select_conf"] = True

    # IRC at DFT level
    if configs["low-irc"] is None or configs["low-irc"].lower()=="true":                                                                                                                              
        configs["low-irc"]= True
    elif configs["low-irc"].lower()=='false':
        configs["low-irc"]= False

    if configs["dft-irc"] is None:
        configs["dft-irc"] = False

    elif configs["dft-irc"].lower() == 'false':
        configs["dft-irc"] = False

    else:
        configs["dft-irc"] = True
    
    # Nimage (default: 9)
    if configs["nimage"] is None:
        configs["nimage"] = 9

    else:
        configs["nimage"] = int(configs["nimage"])

    # add-tor (default: 0.01)
    if configs["add-tor"] is None:
        configs["add-tor"] = 0.01
    else:
        configs["add-tor"] = float(configs["add-tor"])

    # conv-tor (default: 0.0005)
    if configs["conv-tor"] is None:
        configs["conv-tor"] = 0.0005
    else:
        configs["conv-tor"] = float(configs["conv-tor"])

    if configs["eps"] is None:
        configs["eps"]=0.0
    else:
        configs["eps"]=float(configs["eps"])
    if configs["c_method"] is None or configs["c_method"] == "":                                                                                                                                      
        configs["c_method"] = 'crest'

    # Check that solvation option is valid
    if configs["rads"] is None: 
        configs["rads"] = False
    elif configs["rads"].lower() == 'true':
        configs["rads"] = True
    elif configs["rads"].lower() == 'false':
        configs["rads"] = False
    else: configs["rads"] = False
    if configs["rev_rxns"].lower() == 'true':
        configs["rev_rxns"] = True
    elif configs["rev_rxns"].lower() == 'false':
        configs["rev_rxns"] = False
    else: configs["rev_rxns"] = True
    if configs["energy_units"] is None: 
        configs["energy_units"] = 'kcal/mol'

    if configs["kinetics"] is None: 
        configs["kinetics"] = 'gas'
    if configs["reactions"] is None: 
        configs["reactions"] = 'all'
    if configs["energy_units"] is None: 
        configs["energy_units"] = 'kcal/mol'
    if configs["eos"] is None: 
        configs["eos"] = 'ideal-gas'

    if configs["temp"] is None: 
        configs["temp"] = '298'
    if configs["calc_temp"] is None:
        configs["calc_temp"] = '298'
    if configs["pres"] is None: 
        configs["pres"] = '101325'
    if configs["pres_units"] is None: 
        configs["pres_units"] = 'Pa'

    # included to make locate_TS.py still function normally
    if configs["reaction_dict"] is None:
        configs["reaction_dict"] = "None"
    if configs["file_name"] is None:
        configs["file_name"] = 'written_yaml'
    if configs["input_react"] is None:
        configs["input_react"] = 'input_list.txt'
    if configs["input_xyz"] is None:
        configs["input_xyz"] = f"{os.getcwd()}" # gives working folder, won't be any xyz's in there.  Will be replaced anyway

    if configs["config_template"] is None: 
        configs["config_template"] = args.config # if they don't provide a template, copy yaks config (but the file will be crowded with irrelevant info)

    if configs["initial_reactants"] is None:
        print("Please specify an input species in the format (SMILES:1): for example, OC(C=O)CO:1")
        quit()

    if configs["calc_folder"] is None:
        calc_folder =  f"{os.getcwd()}/yaks_output"
        configs["calc_folder"] = calc_folder

    if configs["water_cat"] is None: 
        configs["water_cat"] = False
    elif configs["water_cat"] is True or configs["water_cat"].lower()=="true":
        configs["water_cat"] = True
    else:
        configs["water_cat"] = False

    if configs["ers_queue"] is None: 
        configs["ers_queue"] = 'standby'
    if configs["ers_walltime"] is None: 
        configs["ers_walltime"] = '4'
    if configs['ers_type'].lower() == 'pb3f3':
        configs['ers_type'] = 'pb3f3'
    elif configs['ers_type'].lower() == 'b3f3':
        configs['ers_type'] = 'b3f3'
    else: configs['ers_type'] = 'b2f2'

    if configs["ea_cutoff"] is None: 
        configs["ea_cutoff"] = '65'
    if configs["scratch_path"] is None: 
        configs["scratch_path"] = configs['output_path']
    if configs["scratch_folds"] is None: 
        configs["scratch_folds"] = 'NA'

    if configs["depot_path"] is None: 
        configs["depot_path"] = configs['calc_folder']
    if configs["depot_folds"] is None: 
        configs["depot_folds"] = 'NA'
    if configs["time_step"] is None: 
        configs["time_step"] = '1'
    if configs["steps"] is None: 
        configs["steps"] = '3600'
    if configs["rule"] is None: 
        configs["rule"] = 'css'
    if configs["reactor"] is None: 
        configs["reactor"] = 'IGCPR'
    if configs["num_nodes"] is None: 
        configs["num_nodes"] = int(3)
    if configs["cnf_thresh"] is None: 
        configs["cnf_thresh"] = '6'
    if configs["thresh"] is None:
        configs["thresh"] = 'no'
    if configs["exp_steps"] is None: 
        configs["exp_steps"] = 5
    if configs["uncertainty_guided"] is None: 
        configs["uncertainty_guided"] = False
    elif configs["uncertainty_guided"] == 'true' or configs["uncertainty_guided"] == 'True':
        configs["uncertainty_guided"] = True
    if configs["uncertainty_cycles"] is None:
        configs["uncertainty_cycles"] = 50
    if configs["uncertainty_range"] is None: 
        configs["uncertainty_range"] = 3
    if configs["bimolecular"] is None: 
        configs["bimolecular"] = True
    elif configs["bimolecular"] == 'true' or configs["bimolecular"] == 'True':
        configs["bimolecular"] = True
    else: configs["bimolecular"] = False

    return configs

# Function that sleeps the script until jobids are no longer in a running or pending state in the queue
def monitor_jobs(jobids):
    
    current_jobs = check_queue()
    while True in [ i in current_jobs for i in jobids ]:
        time.sleep(60)
        current_jobs = check_queue()  
    return

# Returns the pending and running jobids for the user as a list
def check_queue():

    # The first time this function is executed, find the user name and scheduler being used. 
    if not hasattr(check_queue, "user"):

        # Get user name
        check_queue.user = subprocess.check_output("echo ${USER}", shell=True).decode('utf-8').strip("\r\n")

        # Get batch system being used
        squeue_tmp = subprocess.Popen(['which', 'squeue'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').strip("\r\n")
        qstat_tmp = subprocess.Popen(['which', 'qstat'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').strip("\r\n")
        check_queue.sched =  None
        if "no squeue in" not in squeue_tmp:
            check_queue.sched = "slurm"
        elif "no qstat in" not in qstat_tmp:
            check_queue.sched = "pbs"
        else:
            print("ERROR in check_queue: neither slurm or pbs schedulers are being used.")
            quit()

    # Get running and pending jobs using the slurm scheduler
    if check_queue.sched == "slurm":

        # redirect a squeue call into output
        output = subprocess.check_output("squeue -l", shell=True).decode('utf-8')

        # Initialize job information dictionary
        jobs = []
        id_ind = None
        for count_i,i in enumerate(output.split('\n')):            
            fields = i.split()
            if len(fields) == 0: continue                
            if id_ind is None and "JOBID" in fields:
                id_ind = fields.index("JOBID")
                if "STATE" not in fields:
                    print("ERROR in check_queue: Could not identify STATE column in squeue -l output.")
                    quit()
                else:
                    state_ind = fields.index("STATE")
                if "USER" not in fields:
                    print("ERROR in check_queue: Could not identify USER column in squeue -l output.")
                    quit()
                else:
                    user_ind = fields.index("USER")
                continue

            # If this job belongs to the user and it is pending or running, then add it to the list of active jobs
            if id_ind is not None and fields[user_ind] == check_queue.user and fields[state_ind] in ["PENDING","RUNNING"]:
                jobs += [fields[id_ind]]

    # Get running and pending jobs using the pbs scheduler
    elif check_queue.sched == "pbs":

        # redirect a qstat call into output
        output = subprocess.check_output("qstat -f", shell=True).decode('utf-8')

        # Initialize job information dictionary
        jobs = []
        job_dict = {}
        current_key = None
        for count_i,i in enumerate(output.split('\n')):
            fields = i.split()
            if len(fields) == 0: continue
            if "Job Id" in i:

                # Check if the previous job belongs to the user and needs to be added to the pending or running list. 
                if current_key is not None:
                    if job_dict[current_key]["State"] in ["R","Q"] and job_dict[current_key]["User"] == check_queue.user:
                        jobs += [current_key]
                current_key = i.split()[2]
                job_dict[current_key] = { "State":"NA", "Name":"NA", "Walltime":"NA", "Queue":"NA", "User":"NA"}
                continue
            if "Job_Name" == fields[0]:
                job_dict[current_key]["Name"] = fields[2]
            if "job_state" == fields[0]:
                job_dict[current_key]["State"] = fields[2]
            if "queue" == fields[0]:
                job_dict[current_key]["Queue"] = fields[2]
            if "Resource_List.walltime" == fields[0]:
                job_dict[current_key]["Walltime"] = fields[2]        
            if "Job_Owner" == fields[0]:
                job_dict[current_key]["User"] = fields[2].split("@")[0]

        # Check if the last job belongs to the user and needs to be added to the pending or running list. 
        if current_key is not None:
            if job_dict[current_key]["State"] in ["R","Q"] and job_dict[current_key]["User"] == check_queue.user:
                jobs += [current_key]

    return jobs

# function to reverse lookup a dictionary key given a value
def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

if __name__ == "__main__":
    main(sys.argv[1:])