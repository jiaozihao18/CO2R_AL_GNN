import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PDB_small
import numpy as np
import ase.io
import os
from ase import neighborlist

def get_mda_atoms(traj_path):

    atoms_l = ase.io.read(traj_path, index=':')

    atoms = atoms_l[0]

    cutOff = neighborlist.natural_cutoffs(atoms)
    cutOff = np.array(cutOff) * 1.5*0.89
    neighborList = neighborlist.NeighborList(cutOff, skin=0, self_interaction=False, bothways=True)
    neighborList.update(atoms)
    conn_matrix = neighborList.get_connectivity_matrix().toarray()

    group_dict = {"slab_top":[*range(24, 36)],
                "slab_bot":[*range(0, 24)]}

    for atom in atoms:
        if atom.symbol == 'O':
            h_indexs = np.where(conn_matrix[atom.index] == 1)[0].tolist()
            group_dict['SOL_%s'%atom.index] = [atom.index] + h_indexs

    resindices = np.zeros(len(atoms))
    for i, key in enumerate(group_dict):
        resindices[group_dict[key]] = i

    u_atoms = mda.Universe.empty(n_atoms=len(atoms),
                                n_residues=len(group_dict),
                                atom_resindex=resindices,
                                trajectory=True) # necessary for adding coordinates

    u_atoms.add_TopologyAttr('resname', ['slab_top', 'slab_bot'] + ['sol']*(len(group_dict)-2))
    u_atoms.add_TopologyAttr('type', list(atoms.symbols))
    u_atoms.add_TopologyAttr('mass', atoms.get_masses())

    bonds = []
    angles = []
    for key, val in group_dict.items():
        if 'SOL' in key:
            bonds.extend([(val[0], val[1]), (val[0], val[2])])
            angles.append([val[1], val[0], val[2]])
            
    u_atoms.add_TopologyAttr('bonds', bonds)
    u_atoms.add_TopologyAttr('angles', angles)

    all_coordinates = [frame.positions[None, :, :] for frame in atoms_l]
    trajectory_coordinates = np.concatenate(all_coordinates, axis=0)

    from MDAnalysis.coordinates.memory import MemoryReader

    u_atoms.load_new(trajectory_coordinates, format=MemoryReader)
    for i, ts in enumerate(u_atoms.trajectory):
        ts.dimensions = atoms_l[i].cell.cellpar()
    
    return u_atoms


u_atoms_ml = get_mda_atoms(traj_path="md_chk15ps_15ps.traj")
u_atoms_dft = get_mda_atoms(traj_path="1cu_h2o/conti3/OUTCAR")

###############################################
from MDAnalysis.analysis import rdf
import matplotlib.pyplot as plt

type_O_ml = u_atoms_ml.select_atoms('type O')
type_H_ml = u_atoms_ml.select_atoms('type H')
type_O_dft = u_atoms_dft.select_atoms('type O')
type_H_dft = u_atoms_dft.select_atoms('type H')

irdf_HH_ml = rdf.InterRDF(type_H_ml, type_H_ml, nbins=75, range=(0.0, 6.0))
irdf_HH_ml.run(start=2000) # 10-15 ps
irdf_HH_dft = rdf.InterRDF(type_H_dft, type_H_dft, nbins=75, range=(0.0, 6.0))
irdf_HH_dft.run(start=5000) # 10-15 ps

irdf_OH_ml = rdf.InterRDF(type_O_ml, type_H_ml, nbins=75, range=(0.0, 6.0))
irdf_OH_ml.run(start=2000) # 10-15 ps
irdf_OH_dft = rdf.InterRDF(type_O_dft, type_H_dft, nbins=75, range=(0.0, 6.0))
irdf_OH_dft.run(start=5000) # 10-15 ps

irdf_OO_ml = rdf.InterRDF(type_O_ml, type_O_ml, nbins=75, range=(0.0, 6.0))
irdf_OO_ml.run(start=2000) # 10-15 ps
irdf_OO_dft = rdf.InterRDF(type_O_dft, type_O_dft, nbins=75, range=(0.0, 6.0))
irdf_OO_dft.run(start=5000) # 10-15 ps

###############################################
fig, axs = plt.subplots(1, 3, figsize=(8, 2.5))

axs[0].plot(irdf_HH_ml.results.bins, irdf_HH_ml.results.rdf, label="ML")
axs[0].plot(irdf_HH_dft.results.bins, irdf_HH_dft.results.rdf, label="DFT")
axs[0].set_xlabel('Radius (angstrom)')
axs[0].set_ylabel('RDF H-H')
axs[0].set_xlim([0.5, 6])
axs[0].set_ylim([0, 10])
axs[0].legend()

axs[1].plot(irdf_OH_ml.results.bins, irdf_OH_ml.results.rdf)
axs[1].plot(irdf_OH_dft.results.bins, irdf_OH_dft.results.rdf)
axs[1].set_xlabel('Radius (angstrom)')
axs[1].set_ylabel('RDF O-H')
axs[1].set_xlim([0.5, 6])
axs[1].set_ylim([0, 10])

axs[2].plot(irdf_OO_ml.results.bins, irdf_OO_ml.results.rdf)
axs[2].plot(irdf_OO_dft.results.bins, irdf_OO_dft.results.rdf)
axs[2].set_xlabel('Radius (angstrom)')
axs[2].set_ylabel('RDF O-O')
axs[2].set_xlim([0.5, 6])
axs[2].set_ylim([0, 10])

plt.tight_layout()