import os, sys, argparse
import matplotlib.pyplot as plt
from utility import *
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole

def main(argv):
    parser=argparse.ArgumentParser(description='Read the IRC_xyz and output a png with energy.')

    #optional arguments
    parser.add_argument('coord_files', help='the folder with multiple IRC pathway or a IRC file.')                                             
    parser.add_argument('-q', dest='q', default=0.0,
                        help = 'The charge state of your reaction')
    args=parser.parse_args()
    args.q=int(args.q)
    if os.path.isdir(args.coord_files):
        inputs = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(args.coord_files) for f in filenames if (fnmatch.fnmatch(f,"*IRC.xyz") )])
    else: inputs=[args.coord_files]
    for _ in inputs:
        elements, geo, energies=read_IRC_pathway(_)
        reactant = geo[0: len(elements)]
        product = geo[len(geo)-len(elements):len(geo)]
        draw_IRC(elements, reactant, product, energies ,namespace=_.split('.xyz')[0],q_tot=args.q)
    return

def read_IRC_pathway(path):
    geo=[]
    energies=[]
    lines=open(path, 'r+').readlines()
    for _ in lines:
        fields=_.split()
        if len(fields)==1:
            elements=["" for i in range(0, (int(fields[0])))]
        if "energy" in _: # The gaussian output
            energies.append(float(fields[-1].split(':')[-1]))
        if "E" in _: # The orca output
            energies.append(float(fields[-1]))
        if len(fields)==4: # the geometry
            for count_i, i in enumerate(elements):
                if i=="":
                    elements[count_i]=fields[0]
                    break
            geo.append([float(fields[1]), float(fields[2]), float(fields[3])])
    e_min=min(energies)
    e=[627.5*(_-e_min) for _ in energies]
    return elements, geo, e 

def draw_IRC(elements, reactant, product, energy ,namespace='plt',q_tot=0):
    smi=return_smi(elements, reactant, q_tot=q_tot)
    R_smi=[_ for _ in smi.split('.')]
    smi=return_smi(elements, product, q_tot=q_tot)
    P_smi=[_ for _ in smi.split('.')]
    R_im=[Chem.MolFromSmiles(_) for _ in R_smi]
    P_im=[Chem.MolFromSmiles(_) for _ in P_smi]
    print("energy barrier from IRC {} is {} kcal/mol.".format(R_smi,max(energy)-energy[0]))
    print("energy barrier from IRC {} is {} kcal/mol.".format(P_smi,max(energy)-energy[-1]))
    fig=plt.figure(figsize=(10,5))
    nodes=range(0, len(energy))
    plt.plot(nodes, energy)
    axes=[0.7, 0.6, 0.2, 0.2]
    opts=Chem.Draw.DrawingOptions()
    opts.bgcolor=None
    for count, _ in enumerate(P_im):
        img=Chem.Draw.MolToImage(_, options=opts)
        axes[0]=axes[0]-0.1*count
        ax=plt.axes(axes, frameon=False)
        ax.imshow(img)
        ax.axis('off')
    axes=[0.1, 0.6, 0.2, 0.2]
    for count, _ in enumerate(R_im):
        img=Chem.Draw.MolToImage(_, options=opts)
        axes[0]=axes[0]+0.1*count
        ax=plt.axes(axes, frameon=False)
        ax.imshow(img)
        ax.axis('off')
    plt.savefig('{}.png'.format(namespace))
    return
main(sys.argv)
