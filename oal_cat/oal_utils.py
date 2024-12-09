from ase.calculators.singlepoint import SinglePointCalculator as sp
import numpy as np
from vasp_interactive import VaspInteractive
from ase.calculators.vasp import Vasp
from ase import neighborlist
from xml.etree import ElementTree
from ase.vibrations.data import VibrationsData
from ase.calculators import calculator
import ase
from ase.constraints import constrained_indices, FixCartesian, FixAtoms


def create_vasp_calc(opt=False, interactive=False, run_cmd="mpirun -np 10 vasp_std", directory='.'):
    common_params = {
        "system": "Cu_alloy", "ncore": 4, "istart": 1, "icharg": 1,
        "lwave": False, "lcharg": False,
        "encut": 500, "ismear": 1, "sigma": 0.2,
        "ediff": 1e-6, "nelmin": 5, "nelm": 60,
        "gga": "RP", "pp": "PBE", "lreal": "Auto",
        "algo": 'Fast', "isym": 0, "ivdw": 11,
        "ediffg": -0.05, "potim": 0.2, "isif": 2,
        "gamma": True, "kpts": [4, 4, 1],
        "command": run_cmd, "directory": directory}
    if opt:
        return Vasp(**common_params, ibrion=2, nsw=2000)
    if interactive:
        return VaspInteractive(**common_params, ibrion=2, nsw=2000)
    else:
        return Vasp(**common_params, ibrion=-1, nsw=0)

def create_vasp_calc_freq(run_cmd="mpirun -np 10 vasp_std", directory='.'):
    common_params = {
        "system": "Cu_alloy", "ncore": 4, "istart": 1, "icharg": 1,
        "lwave": False, "lcharg": False,
        "encut": 500, "ismear": 1, "sigma": 0.2,
        "ediff": 1e-7, "nelmin": 5, "nelm": 60,
        "gga": "RP", "pp": "PBE", "lreal": "Auto",
        "algo": 'Fast', "isym": 0, "ivdw": 11,
        "ediffg": -0.05, "potim": 0.015, "isif": 2,
        "gamma": True, "kpts": [4, 4, 1],
        "command": run_cmd, "directory": directory,
        "ibrion": 5, "nsw": 1, "nfree": 2}

    return Vasp(**common_params)

def write_db(atoms_l, db):
    for atoms in atoms_l:
        db.write(atoms)

def parent_only_replay(calc, optimizer):
    """Reinitialize hessian when there is a parent call based on certain criteria."""
    if calc.info.get("check", False):
        
        dataset = calc.parent_dataset
        
        optimizer.H = None
        atoms = dataset[0]
        r0 = atoms.get_positions().ravel()
        f0 = atoms.get_forces(apply_constraint=False).ravel()
        # for eligible atoms added to dataset, update the hessian using the replay function
        for atoms in dataset:
           
            # pass both the base atoms and atoms with the ml calc in case replay function wants either
            r = atoms.get_positions().ravel()
            f = atoms.get_forces(apply_constraint=False).ravel()
            
            optimizer.update(r, f, r0, f0)
            r0 = r
            f0 = f

        optimizer.r0 = dataset[-1].get_positions().ravel()
        optimizer.f0 = dataset[-1].get_forces(apply_constraint=False).ravel()

def copy_images(images, with_calc=True):
    """
    images: list
        List of ase atoms images to be copied.
    """
    new_images = []
    for image in images:
        new_image = image.copy()
        if with_calc and image.get_calculator():
            calc = image.get_calculator()
            new_image.set_calculator(calc)
        new_images.append(new_image)
    return new_images

def convert_to_sp(images):
    """
    Replaces the attached calculators with singlepoint calculators

    images: list
        List of ase atoms images with attached calculators for forces and energies.
    """
    images = copy_images(images, with_calc=True)
    singlepoint_images = []
    
    for image in images:
        if isinstance(image.get_calculator(), sp):
            singlepoint_images.append(image)
            continue
        
        sample_energy = image.get_potential_energy(apply_constraint=False)
        sample_forces = image.get_forces(apply_constraint=False)
        
        sp_calc = sp(atoms=image, energy=float(sample_energy), forces=sample_forces)
        sp_calc.implemented_properties = ["energy", "forces"]
        image.set_calculator(sp_calc)
        
        singlepoint_images.append(image)
       
    return singlepoint_images

def get_atoms_calc(images, calc):
    
    images = copy_images(images, with_calc=False)
    for image in images:
        image.set_calculator(calc)
        
    images_calc = convert_to_sp(images)
    
    return images_calc

def get_error(atoms, atoms_pre):
    
    forces = atoms.get_forces()
    forces_pre = atoms_pre.get_forces()
    
    ene = atoms.get_potential_energy(apply_constraint=False)
    ene_pre = atoms_pre.get_potential_energy(apply_constraint=False)
    ene_error = ene - ene_pre
    
    if atoms.constraints:
        constraints_index = atoms.constraints[0].index
    else:
        constraints_index = []
        
    forces_error = np.sum(np.abs(np.delete(forces - forces_pre, constraints_index, axis=0)))
    
    relative_forces_error = np.divide(np.sum(np.abs(np.delete(forces - forces_pre, constraints_index, axis=0))),
                                      np.sum(np.abs(np.delete(forces, constraints_index, axis=0)))).item()
    
    result = {}
    result["ene_error"] = ene_error
    result["forces_error"] = forces_error
    result["relative_forces_error"] = relative_forces_error
    
    return result


def is_up_then_down(arr):
    arr = np.array(arr)
    max_index = np.argmax(arr)
    if max_index == 0 or max_index == len(arr) - 1:
        return False
    if not np.all(arr[:max_index] <= arr[1:max_index + 1]):
        return False
    if not np.all(arr[max_index:-1] >= arr[max_index + 1:]):
        return False
    return True


def get_matrix(atoms, ratio=1.5*0.89):
    
    cutOff = neighborlist.natural_cutoffs(atoms)
    cutOff = np.array(cutOff) * ratio
    neighborList = neighborlist.NeighborList(cutOff, skin=0,
                                             self_interaction=False, bothways=True)
    neighborList.update(atoms)
    matrix = neighborList.get_connectivity_matrix().toarray()

    return matrix

def get_neighbors(atoms, index, ratio=1.5*0.89):
    """
    :param atoms:
    :param index:
    :return:
    """
    cutOff = neighborlist.natural_cutoffs(atoms)
    cutOff = np.array(cutOff) * ratio
    neighborList = neighborlist.NeighborList(cutOff, skin=0,
                                             self_interaction=False, bothways=True)
    neighborList.update(atoms)
    neigh = neighborList.get_neighbors(index)[0]
    return neigh

def dfs(adjMatrix, vertex, visited):
    visited[vertex] = True
    for i in range(len(adjMatrix)):
        if adjMatrix[vertex][i] == 1 and not visited[i]:
            dfs(adjMatrix, i, visited)
            

def isConnected(adjMatrix):
    n = len(adjMatrix)
    visited = [False] * n
    dfs(adjMatrix, 0, visited)
    return all(visited)


def read_massweighted_hessian_xml() -> np.ndarray:
    """Read the Mass Weighted Hessian from vasprun.xml.

    Returns:
        The Mass Weighted Hessian as np.ndarray from the xml file.

        Raises a ReadError if the reader is not able to read the Hessian.

        Converts to ASE units for VASP version 6.
    """

    file = 'vasprun.xml'
    try:
        tree = ElementTree.iterparse(file)
        hessian = None
        for event, elem in tree:
            if elem.tag == 'dynmat':
                for i, entry in enumerate(elem.findall('varray[@name="hessian"]/v')):
                    text_split = entry.text.split()
                    if not text_split:
                        raise ElementTree.ParseError("Could not find varray hessian!")
                    if i == 0:
                        n_items = len(text_split)
                        hessian = np.zeros((n_items, n_items))
                    assert isinstance(hessian, np.ndarray)
                    hessian[i, :] = np.array([float(val) for val in text_split])
                if i != n_items - 1:
                    raise ElementTree.ParseError("Hessian is not quadratic!")
                #VASP6+ uses THz**2 as unit, not mEV**2 as before
                for entry in elem.findall('i[@name="unit"]'):
                    if entry.text.strip() == 'THz^2':
                        conv = ase.units._amu / ase.units._e / 1e-4 * (2 * np.pi)**2  # THz**2 to eV**2
                        # VASP6 uses factor 2pi
                        # 1e-4 = (angstrom to meter times Hz to THz) squared = (1e10 times 1e-12)**2
                        break
                    else:  # Catch changes in VASP
                        vasp_version_error_msg = (
                            f'The file "{file}" is from a non-supported VASP version. '
                            'Not sure what unit the Hessian is in, aborting.')
                        raise calculator.ReadError(vasp_version_error_msg)

                else:
                    conv = 1.0  # VASP version <6 unit is meV**2
                assert isinstance(hessian, np.ndarray)
                hessian *= conv
        if hessian is None:
            raise ElementTree.ParseError("Hessian is None!")

    except ElementTree.ParseError as exc:
        incomplete_msg = (
            f'The file "{file}" is incomplete, and no DFT data was available. '
            'This is likely due to an incomplete calculation.')
        raise calculator.ReadError(incomplete_msg) from exc
    # VASP uses the negative definition of the hessian compared to ASE
    return -hessian


def get_vibrations(calc) -> VibrationsData:
    """Get a VibrationsData Object from a VASP Calculation.

    Returns:
        VibrationsData object.

        Note that the atoms in the VibrationsData object can be resorted.

        Uses the (mass weighted) Hessian from vasprun.xml, different masses
        in the POTCAR can therefore result in different results.

        Note the limitations concerning k-points and symmetry mentioned in
        the VASP-Wiki.
    """

    mass_weighted_hessian = read_massweighted_hessian_xml()
    #get indices of freely moving atoms, i.e. respect constraints.
    
    const_indices = constrained_indices(calc.atoms, only_include=(FixCartesian, FixAtoms))
    #Invert the selection to get free atoms
    indices = np.setdiff1d(np.array(range(len(calc.atoms))), const_indices).astype(int)

    #save the corresponding sorted atom numbers
    sort_indices = np.array(calc.sort)[indices]
    #mass weights = 1/sqrt(mass)
    mass_weights = np.repeat(calc.atoms.get_masses()[sort_indices]**-0.5, 3)
    #get the unweighted hessian = H_w / m_w / m_w^T
    #ugly and twice the work, but needed since vasprun.xml does not have the unweighted
    #ase.vibrations.vibration will do the opposite in Vibrations.read
    hessian = mass_weighted_hessian / mass_weights / mass_weights[:, np.newaxis]

    return VibrationsData.from_2d(calc.atoms[calc.sort], hessian, indices)


def adjust_structure_after_freq_calc(calc, threshold=100):
    
    vib = get_vibrations(calc)

    img_freq_idx = []
    img_freq_val = []
    for i, fre in enumerate(vib.get_frequencies()):
        if fre.real == 0 and fre.imag > threshold:
            img_freq_idx.append(i)
            img_freq_val.append(fre)

    if len(img_freq_idx) != 1:
        print("No single significant imaginary frequency found, img_freq_val:%s."%img_freq_val)
        return None

    print("Found the only imaginary frequency, img_freq_val:%s."%img_freq_val)
    atoms = vib.get_atoms()

    # Normalize the displacement factor
    f = 0.5 / np.linalg.norm(vib.get_modes()[img_freq_idx[0]], axis=1).sum()

    atoms_cp = atoms.copy()
    atoms_cp.positions[vib.get_mask()] += vib.get_modes()[img_freq_idx[0]] * f

    atoms_cm = atoms.copy()
    atoms_cm.positions[vib.get_mask()] -= vib.get_modes()[img_freq_idx[0]] * f

    # Choose the structure based on carbon atom distances
    if atoms_cp.get_distance(*np.where(atoms_cp.symbols == 'C')[0]) > \
       atoms_cm.get_distance(*np.where(atoms_cm.symbols == 'C')[0]):
        return atoms_cp, atoms_cm
    else:
        return atoms_cm, atoms_cp