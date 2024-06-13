import sys
import os
import fnmatch
from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers, AllChem, TorsionFingerprints, rdmolops, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.ML.Cluster import Butina
from itertools import permutations
import argparse
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
from taffi_functions import *
def gen_rdkit_conf(mol, nconf=20, attempts=1000000, thresh=1.0, torsion_angle=False, knowledge=True, chirality=False):
    ids=AllChem.EmbedMultipleConfs(mol, useRandomCoords=True,numConfs=nconf, maxAttempts=attempts, pruneRmsThresh=thresh, useExpTorsionAnglePrefs=torsion_angle,\
                                   useBasicKnowledge=knowledge, enforceChirality=chirality)
    return mol, list(ids)
def conf_rdkit(input_xyz, output_path):
    input_E, input_G=xyz_parse(input_xyz)
    input_table=Table_generator(input_E, input_G)
    # write mol file
    print(input_xyz)
    mol_write("{}.mol".format(input_xyz.split('.')[0]),input_E,input_G, input_table)
    # convert mol file into rdkit mol object
    mol=Chem.rdmolfiles.MolFromMolFile('{}.mol'.format((input_xyz.split('.')[0])),removeHs=False)
    os.system('rm {}.mol'.format((input_xyz.split('.')[0])))
    isomers=tuple(EnumerateStereoisomers(mol))
    smiles=[smi for smi in sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers)]
    if os.path.isdir('{}/{}'.format(output_path,input_xyz.split('/')[-1].split('.')[0])) is False:
        os.system('mkdir {}/{}'.format(output_path,input_xyz.split('/')[-1].split('.')[0]))
    output_folder=output_path+'/'+input_xyz.split('/')[-1].split('.')[0]
    if os.path.isdir('{}/results'.format(output_folder)) is False: os.system('mkdir {}/results'.format(output_folder))
    count=0
    for count_mo, mo in enumerate(isomers):
        if count_mo>0: count=count+len(ids)
        mo, ids=gen_rdkit_conf(mo)
        for i in ids:
            Chem.rdmolfiles.MolToXYZFile(mo, '{}/results/{}.xyz'.format(output_folder, count+i), confId=i)
    xyz_files=[output_folder+'/results/'+i for i in os.listdir('{}/results'.format(output_folder)) if fnmatch.fnmatch(i, '*.xyz')]
    for i in xyz_files:
        E, G= xyz_parse(i)
        table=Table_generator(E, G)
        if table.all()!=input_table.all(): os.system('rm {}'.format(i))
