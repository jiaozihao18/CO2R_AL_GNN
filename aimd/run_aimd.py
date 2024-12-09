import numpy as np
from ase import Atoms
import ase.io
from ase.calculators.vasp import Vasp

def gen_slab_water(slab, h=3, layers=3):
    p = np.array(
        [[ 0.   ,  0.   ,  1.288],
         [ 0.64 , -0.948,  1.236],
         [ 0.641,  0.948,  1.233],
         [ 1.578,  2.451,  1.385],
         [ 2.732,  2.441,  1.254],
         [ 1.498,  2.449,  2.486],
         [ 4.244,  2.43 ,  1.195],
         [ 4.891,  1.481,  1.248],
         [ 4.878,  3.378,  1.252],
         [ 5.824,  0.003,  1.101],
         [ 6.928,  0.003,  1.229],
         [ 5.741, -0.009,  0.   ]])
    c = np.array([[8.490, 0., 0.],
                [0., 4.902, 0.],
                [0., 0., 2.9]])
    W = Atoms('4(OH2)', positions=p, cell=c, pbc=[1, 1, 1])
    W = W*[1, 2, layers]
    # m = sum(W.get_masses())/units.mol # g
    # v = W.get_volume()*1e-24 # cm3
    # rio = m/v
    # print("density is %s"%rio)

    # expand W only in xy direction not z
    W.set_cell(np.vstack([slab.cell[:2, :], W.cell[2, :]]), scale_atoms=True)
    disp = slab.positions[:, 2].max() - W.positions[:, 2].min() + h
    W.positions[:, 2] += disp
    
    slab += W
    slab.center(10, axis=2)
    
    return slab

def create_vasp_calc(run_cmd="mpirun -np 10 vasp_std", directory='.'):
    common_params = {
        "system": "Cu_alloy", "ncore": 4, "istart": 1, "icharg": 1,
        "lwave": False, "lcharg": False,
        "encut": 400, "ismear": 1, "sigma": 0.2,
        "ediff": 1e-4, "nelmin": 5, "nelm": 60,
        "gga": "RP", "pp": "PBE", "lreal": "Auto",
        "algo": 'Fast', "isym": 0, "ivdw":11,
        
        "ibrion": 0, "nsw":500, "potim": 1, 
        "smass": 0.5, "mdalgo":2, "tebeg":298, "teend":298,
        
        "gamma": True, "kpts": [3, 2, 1],
        "command": run_cmd, "directory": directory}
    
    return Vasp(**common_params)

atoms = ase.io.read("vasp_slab/OUTCAR")
atoms = gen_slab_water(atoms, h=3, layers=3)

calc = create_vasp_calc(run_cmd="mpirun vasp_std", directory='./')
atoms.calc = calc

calc.calculate(atoms)
