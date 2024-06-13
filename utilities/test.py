from utility import *
from taffi_functions import *
E,G = xyz_parse('/scratch/bell/hsu205/reverse_EC/low-IRC-result/CILLQUBPNHRFCU_10_0_2-end.xyz')
adj=Table_generator(E,G)
print(return_smi(E,G))
print(return_smi(E,G,adj))
print(E)
print(adj)
