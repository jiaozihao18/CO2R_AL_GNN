from my_utils import *
from ase.db import connect
from ase.calculators.vasp import Vasp
from ase.io.trajectory import Trajectory
import ase.io
import os, sys

file_name = sys.argv[1]

atoms_sl = ase.io.read(file_name, ':')

os.makedirs("./traj", exist_ok=True)

for i, atoms_s in enumerate(atoms_sl):
    
    if os.path.isfile("./traj/%s.traj"%i):
        continue
    
    add_water(atoms_s, get_water_unit(), 2.5)
    calc = Vasp()
    
    calc.set(gamma=True, kpts=[3, 3, 1])
    calc.set(ncore=4, istart=1, icharg=1,
             lwave=0, lcharg=0,
             #SCF
             encut=450, ismear=1, sigma=0.2,
             ivdw=11, # ediff=1e-4,
             gga='rp', pp='pbe', lreal='auto',
             algo='Fast', isym=0,
             #GEO OPT
             ediffg=-0.05, ibrion=2, # potim=0.5,
             nsw=50, isif=0,
             )
    
    atoms_s.calc=calc
    calc.calculate(atoms_s)
    
    tags = atoms_s.get_tags()
    sort_l, resort_l = read_sort('./ase-sort.dat')
    tags = tags[sort_l]
    
    traj = Trajectory("./traj/%s.traj"%i, 'w')
    for tmp in ase.io.read('OUTCAR', ':'):
        tmp.set_tags(tags)
        traj.write(tmp)
    traj.close()
    
    
    
    