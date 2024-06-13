import os, sys
from utility import *
from xyz_to_Gaussian import *
from job_submit import *
from taffi_functions import *

def main(argv):
    inf=open(argv[1], 'r+')
    xyz_folder='/'.join(argv[1].split('/')[:-1])+'/xyz_folder'
    #print(xyz_folder)
    if os.path.exists(xyz_folder) is False: os.mkdir(xyz_folder)
    lines=inf.readlines()
    gjf_path=[]
    smiles=[]
    NE=[]
    inf.close()
    for count_i, i in enumerate(lines):
       fields= i.split()
       argu, E, G, q=parse_smiles(fields[0], ff='mmff94')
       smiles.append(fields[0])
       NE.append(len(E))
       if argu:
           xyz_file='/'.join(argv[1].split('/')[:-1])+'/xyz_folder/{}'.format(count_i)
           adj_mat = Table_generator(E,G)
           lone_electrons,bonding_electrons,core_electrons,bond_mat,fc = find_lewis(E,adj_mat,q_tot=q,keep_lone=[],return_pref=False,return_FC=True)
           keep_lone = [ [ count_j for count_j,j in enumerate(lone_electron) if j%2 != 0] for lone_electron in lone_electrons][0]
           multiplicity = 1+len(keep_lone)
           xyz_write(xyz_file+'.xyz', E,G)
           substring = "python {}/utilities/xyz_to_Gaussian.py {}.xyz -o {}.gjf -q {} -m {} -c \"{}\" -ty \"{}\" -t \"{} DFT\""
           #substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),xyz_file,xyz_file,q,multiplicity,"False",\
           #                             "wB97X-D/6-31+G* Opt=(maxcycles=100) Int=UltraFine SCF=QC",fields[0])
           substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),xyz_file,xyz_file,0, multiplicity,"False",\
                                        "B3LYP/Def2TZVPP EmpiricalDispersion=GD3 Opt=(maxcycles=10000) Int=UltraFine SCF=QC Freq",fields[0])
           output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
           os.system(substring)
           gjf_path+=['{}.gjf'.format(xyz_file)]
    substring="python {}/utilities/Gaussian_submit.py -f '*.gjf' -ff \"{}\" -d {} -para perpendicular -p 1 -n 1 -ppn 24  -q bsavoie -mem 2000 -sched slurm -t 4 -o G_opt --silent"
    substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),gjf_path, xyz_folder)   
    output=subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
    #substring="python {}/utilities/job_submit.py -f 'G_opt.*.submit' -sched slurm".format('/'.join(os.getcwd().split('/')[:-1]))
    #output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
    #monitor_jobs(output.split())
    #os.system('rm *.submit')
    out=open('energy.txt','w+')
    for count_i, i in enumerate(smiles):
        file_name=xyz_folder+'/{}.out'.format(count_i)
        xyz_name=xyz_folder+'/{}.xyz'.format(count_i)
        finish,_,_,_,_,energy,_=read_Gaussian_output(file_name)
        if finish:
            out.write('{} {} {}\n'.format(count_i, i, energy))
            substring='python {}/utilities/read_Gaussian_output.py -i {} -o {} -n {} --count'
            substring=substring.format('/'.join(os.getcwd().split('/')[:-1]), file_name, xyz_name, NE[count_i])
            subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        else:
            out.write('{} {} N/A\n'.format(count_i, i))
    out.close()
    return
main(sys.argv)
