import numpy as np
from ase import Atoms
import ase.io
import sys
from oal_cat.oal_utils import *
from oal_cat.run_task import run_calc
from ase.optimize.minimahopping import MinimaHopping
from ase.constraints import FixAtoms, Hookean
import os


def gen_slab_water(slab, h=3):
    p = np.array(
        [[0.27802511, -0.07732213, 13.46649107],
        [0.91833251, -1.02565868, 13.41456626],
        [0.91865997, 0.87076761, 13.41228287],
        [1.85572027, 2.37336781, 13.56440907],
        [3.00987926, 2.3633134, 13.4327577],
        [1.77566079, 2.37150862, 14.66528237],
        [4.52240322, 2.35264513, 13.37435864],
        [5.16892729, 1.40357034, 13.42661052],
        [5.15567324, 3.30068395, 13.4305779],
        [6.10183518, -0.0738656, 13.27945071],
        [7.2056151, -0.07438536, 13.40814585],
        [6.01881192, -0.08627583, 12.1789428]])
    c = np.array([[8.490373, 0., 0.],
                [0., 4.901919, 0.],
                [0., 0., 24.1]])
    W = Atoms('4(OH2)', positions=p, cell=c, pbc=[1, 1, 1])
    W = W*[1, 2, 1]

    W.set_cell(slab.cell, scale_atoms=True)
    disp = slab.positions[:, 2].max() - W.positions[:, 2].min() + h
    W.positions[:, 2] += disp
    W.positions[:, 0] -= 0.278
    
    return slab + W

def run_cmh(atoms, calc):
    
    # zmax = atoms.positions[:, 2].max()
    constraints = atoms.constraints
    for index1 in range(len(atoms)):
        if atoms[index1].symbol == 'O':
            # constraints.append(Hookean(a1=index1, a2=(0., 0., 1., -zmax), k=5.))
            for index2 in get_neighbors(atoms, index1):
                if atoms[index2].symbol == 'H':
                    constraints.append(Hookean(a1=index1, a2=int(index2), rt=1.4, k=5))          
    # constraints.append(FixAtoms(indices=[atom.index for atom in atoms if atom.symbol not in ['H', 'O']]))
    atoms.set_constraint(constraints)

    with calc:
        atoms.calc = calc
        hop = MinimaHopping(atoms,
                            Ediff0=2.5,
                            T0=1000.,
                            fmax=0.05)
        hop(totalsteps=30)



atoms = ase.io.read("vasp/vasp_slab/OUTCAR")
atoms = gen_slab_water(atoms, h=3)
calc = create_vasp_calc(interactive=True, run_cmd="mpirun vasp_std")

os.makedirs("vasp/cmh", exist_ok=True)
os.chdir("vasp/cmh")
run_cmh(atoms, calc)