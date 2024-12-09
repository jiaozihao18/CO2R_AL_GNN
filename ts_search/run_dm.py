from ase.db import connect
import numpy as np
import os
import sys
from oal_cat.run_task import run_dm, run_calc, run_oal_opt
from oal_cat.oal_utils import *
from oal_cat.mlCalculator import mlCalculator
import ase.io

# remenber consider the index sort
# atoms = ase.io.read("POSCAR_pos")
# calc = create_vasp_calc(opt=True, run_cmd="mpirun vasp_std", directory='./fin_opt')
# run_calc(atoms, calc)

########### Cs1 not move, Cs2 is "CO" which move, C index first
Cs1 = [64, 44]
Cs2 = [63, 45]

atoms = ase.io.read("./fin_opt/OUTCAR", index=-1)
checkpoint_path = "gemnet_t_direct_h512_all.pt"
ml_calc = mlCalculator(checkpoint_path=checkpoint_path, cpu=False)
vasp_calc = create_vasp_calc(run_cmd="mpirun vasp_std", directory='./oal_dimer', interactive=False)
run_dm(atoms, Cs1[0], Cs2[0], Cs2, ml_calc, vasp_calc)

atoms = ase.io.read("dimer.db", index=-1)
freq_calc = create_vasp_calc_freq(run_cmd="mpirun vasp_std", directory='./freq_vasp')
run_calc(atoms, freq_calc, fix_index=[atom.index for atom in atoms if atom.index not in Cs1+Cs2])

os.chdir("./freq_vasp")
calc = Vasp(restart=True)
atoms_ini, atoms_fin = adjust_structure_after_freq_calc(calc)
os.chdir("../")
if atoms_ini:
    calc = create_vasp_calc(opt=True, run_cmd="mpirun vasp_std", directory='./ini_from_freq')
    run_calc(atoms_ini, calc, fix_index=[*range(12)])
if atoms_fin:
    calc = create_vasp_calc(opt=True, run_cmd="mpirun vasp_std", directory='./fin_from_freq')
    run_calc(atoms_fin, calc, fix_index=[*range(12)])
    

