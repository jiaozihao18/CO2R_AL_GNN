from ase import neighborlist
from ase.data import covalent_radii as CR
import numpy as np
import ase.io
from ase.visualize import view
from numpy import array
from ase import Atoms
import os

def read_sort(file_path):
    # 'ase-sort.dat'
    sort_l = []
    resort_l = []
    with open(file_path, 'r') as fd:
        for line in fd:
            sort, resort = line.split()
            sort_l.append(int(sort))
            resort_l.append(int(resort))
    return sort_l, resort_l

def get_water_unit():
    """
    structure from constrained minima hopping
    """
    water_dict = {'numbers': array([8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1]),
                'positions': array([[ 0.50095307,  0.12765577, 14.30718825],
                        [ 1.75429364,  2.45080683, 13.50431975],
                        [ 4.75504138,  2.638159  , 13.27097652],
                        [ 5.9930533 ,  0.31031824, 14.08464408],
                        [ 0.95542756,  0.96892179, 14.01225124],
                        [ 0.63326272,  0.11064232, 15.27074338],
                        [ 2.73189671,  2.55143079, 13.49180379],
                        [ 1.3823925 ,  3.33750965, 13.70981876],
                        [ 5.20601734,  1.79402858, 13.56312415],
                        [ 4.91117065,  2.67646334, 12.31222678],
                        [ 5.62626144, -0.58005093, 13.8820935 ],
                        [ 6.97109682,  0.21912052, 14.07967383]]),
                'cell': array([[ 8.490373,  0.      ,  0.      ],
                        [ 0.      ,  4.901919,  0.      ],
                        [ 0.      ,  0.      , 26.93236 ]]),
                'pbc': array([ True,  True,  True])}
    atoms_w = Atoms.fromdict(water_dict)
    atoms_w.set_tags(3)
    return atoms_w

def get_cm(atoms):
    """
    criterion for ase gui
    """
    cutOff = neighborlist.natural_cutoffs(atoms)
    cutOff = np.array(cutOff) * 1.5*0.89 
    neighborList = neighborlist.NeighborList(cutOff, skin=0,
                                            self_interaction=False, bothways=True)
    neighborList.update(atoms)

    cm = neighborList.get_connectivity_matrix()
    return cm

def filter_outer(atoms):
    """
    filter outer and avoid interaction between images
    """
    atoms.pbc = False
    sp = atoms.get_scaled_positions()
    cm = get_cm(atoms)
    
    o_l = [atom.index for atom in atoms if atom.symbol == 'O' 
           and np.all(sp[atom.index]<=1)  
           and np.all(sp[atom.index]>=0)]
    
    s_l = []
    for o_i in o_l:
        h_i = list(cm[o_i].nonzero()[1])
        atoms_tmp = atoms[s_l+[o_i] + h_i]
        atoms_tmp.pbc = True
        # add one water means add 4 bonds
        if get_cm(atoms_tmp).sum() == len(s_l)/3*4 + 4:
            s_l.extend([o_i] + h_i)
                
    atoms_f = atoms[s_l]
    atoms_f.pbc = True
            
    return atoms_f

def add_water(atoms_s, w_unit, h):
    
    normal_v = np.cross(atoms_s.cell.array[0], atoms_s.cell.array[1])
    atoms_s.rotate(normal_v, 'z')

    # make water supercell
    ss = atoms_s.positions.ptp(axis=0)
    ww = w_unit.positions.ptp(axis=0)
    [m, n, _] = np.floor_divide(ss, ww) + 1
    atoms_w = w_unit.repeat((int(m), int(n), 1))
    
    atoms_w.positions[:,2] += (-atoms_w.positions[:,2].min()
                            + atoms_s.positions[:,2].max()
                            + h)
    atoms_s.rotate('z', normal_v)
    
    atoms_w.cell = atoms_s.cell
    atoms_w.rotate('z', normal_v)
    
    atoms_w.pbc = False
    scaled_pos_min = atoms_w.get_scaled_positions().min(axis=0)
    scaled_pos = atoms_w.get_scaled_positions()
    scaled_pos[:, 0] -= scaled_pos_min[0]
    scaled_pos[:, 1] -= scaled_pos_min[1]
    atoms_w.set_scaled_positions(scaled_pos)
    atoms_w.pbc = True
    
    atoms_f = filter_outer(atoms_w)
    
    atoms_s.extend(atoms_f)