from .mlCalculator import mlCalculator
from .oalCalculator import oalCalculator
import ase.io
from ase.optimize import BFGS, FIRE
import numpy as np
from ase.visualize import view
import os
from .oal_utils import create_vasp_calc, get_atoms_calc, create_vasp_calc_freq, parent_only_replay, is_up_then_down
from ase.calculators.vasp import Vasp
from ase.calculators.emt import EMT
from ase.db import connect
import time
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.constraints import FixAtoms
from ase.neb import NEB
import copy
from ase.data import covalent_radii as CR


def run_oal_opt(atoms, ml_calc, vasp_calc, name_prefix="oal", qn=50, fmax=0.05, steps=500):
    start = time.time()
    calc = oalCalculator(ml_calc=ml_calc, 
                         parent_calc=vasp_calc,
                         initial_points=1,
                         query_every_n_steps=qn,
                         fmax_verify_threshold=fmax,
                         partial_fit_num=1,
                         train_epochs=400,
                         oal_db_name="%s.db"%name_prefix)
    atoms.calc = calc
    dyn = BFGS(atoms, trajectory="%s.traj"%name_prefix)
    dyn.attach(parent_only_replay, 1, calc, dyn)
    dyn.run(fmax=fmax, steps=steps)
    
    end = time.time()
    print("=== Time using: %s seconds for opt!"%(end - start), flush=True)


def get_displacement_vector(atoms, idx1, idx2, m=0.5):
   
    p1 = atoms.positions[idx1][:2]
    p2 = atoms.positions[idx2][:2]
    d = p2 - p1
    unit_d = d / np.linalg.norm(d)
    dv = unit_d * m
    dv = np.append(dv, 0)
    
    return dv

def write_dm_db(dyn, db, loop_num):
    atoms_ = dyn.atoms.atoms
    atoms_ = get_atoms_calc([atoms_], atoms_.calc)[0]
    db.write(atoms_, loop_num=loop_num)

def spy_curvature(dyn, save_l):
    save_l.append(dyn.atoms.get_curvature())
    if dyn.atoms.get_curvature() > 10 and dyn.nsteps > 0:
        dyn.max_steps = dyn.nsteps
        

def run_dm(atoms, c1_idx, c2_idx, displace_idxs, ml_calc, vasp_calc, steps_num=10, reload_chk=False, train_all=False):
    
    """
    displacement vector point form c1_idx to c2_idx
    """
    
    start = time.time()
    db = connect("dimer.db", append=False)
    db_all = connect("dimer_all.db", append=False)
    image_dft_l = []

    # pre-train
    image_ml = get_atoms_calc([atoms], ml_calc)[0]
    image_dft = get_atoms_calc([atoms], vasp_calc)[0]
    image_dft_l.append(image_dft)
    ml_calc.train([image_dft], epoches=400, batch_size=1, reload_chk=reload_chk)
    image_ml_retrain = get_atoms_calc([atoms], ml_calc)[0]
    db.write(image_dft, 
             ml_ene=image_ml.get_potential_energy(),
             ml_fmax=np.sqrt(np.sum(image_ml.get_forces()**2, axis=1)).max(),
             ml_ene_retrain=image_ml_retrain.get_potential_energy(),
             ml_fmax_retrain=np.sqrt(np.sum(image_ml_retrain.get_forces()**2, axis=1)).max().item())
    
    atoms.calc = ml_calc
    
    dist_min_threshhold = 0.9*(CR[atoms.numbers[c1_idx]] + CR[atoms.numbers[c2_idx]]) #0.76*2*0.9=1.368
    dist_max_threshhold = 2*dist_min_threshhold # 3.5
    
    print("dist_min_threshhold: %s, dist_max_threshhold: %s"%(dist_min_threshhold, dist_max_threshhold), flush=True)
    
    dv = get_displacement_vector(atoms, c1_idx, c2_idx, 0.2*dist_min_threshhold)
    if atoms.constraints:  
        mask = [0 if i in atoms.constraints[0].index else 1 for i in range(len(atoms))]
    else:
        mask=None
    d_control = DimerControl(initial_eigenmode_method='displacement',
                             displacement_method='vector', 
                             logfile=None,
                             mask=mask)
    d_atoms = MinModeAtoms(atoms, d_control)
    displacement_vector = np.zeros((len(atoms), 3))
    displacement_vector[displace_idxs] += dv
    d_atoms.displace(displacement_vector=displacement_vector)
    
    old_eigenmodes = copy.deepcopy(d_atoms.eigenmodes)
    old_curvatures = copy.deepcopy(d_atoms.curvatures)
    old_atoms = d_atoms.atoms.copy()
    
    steps = steps_num
    loop_num = 0
    while True:
        loop_num += 1
        
        d_atoms.check_atoms = None
        dim_rlx = MinModeTranslate(d_atoms, trajectory=None, logfile='-')
        dim_rlx.attach(write_dm_db, 1, dim_rlx, db_all, loop_num)
        save_l = []
        dim_rlx.attach(spy_curvature, 1, dim_rlx, save_l)
        dim_rlx.run(fmax=0.05, steps=steps)
        
        dist = d_atoms.atoms.get_distance(c1_idx, c2_idx, mic=True)
        if dist > dist_max_threshhold or dist < dist_min_threshhold or min(save_l)>0:
            if dist > dist_max_threshhold or dist < dist_min_threshhold:
                print("=== %s Exceed min max threshhold, retrain and run from the last frame"%loop_num, flush=True)
                steps = 1
            else:
                print("=== %s min(save_l)>0, retrain and run from the last frame"%loop_num, flush=True)
                steps = steps_num
                
            image_dft = get_atoms_calc([d_atoms.atoms], vasp_calc)[0]
            db.write(image_dft)
            # if steps > 5:
            #     steps -= 5
            # ml_calc.train([*image_dft_l[-5:], image_dft], epoches=100, batch_size=len(image_dft_l[-5:])+1, reload_chk=reload_chk)
                
            ml_calc.train([image_dft], epoches=400, batch_size=1, reload_chk=reload_chk)
            
            d_atoms.atoms = old_atoms.copy()
            d_atoms.eigenmodes = copy.deepcopy(old_eigenmodes)
            d_atoms.curvatures = copy.deepcopy(old_curvatures)
            d_atoms.atoms.calc = ml_calc
            
        else:
            old_eigenmodes = copy.deepcopy(d_atoms.eigenmodes)
            old_curvatures = copy.deepcopy(d_atoms.curvatures)
            old_atoms = d_atoms.atoms.copy()
            
            image_ml = get_atoms_calc([d_atoms.atoms], ml_calc)[0]
            image_dft = get_atoms_calc([d_atoms.atoms], vasp_calc)[0]
            image_dft_l.append(image_dft)
            
            fmax_dft = np.sqrt(np.sum(image_dft.get_forces()**2, axis=1)).max()
            print("=== %s fmax_dft = %s"%(loop_num, fmax_dft), flush=True)
            if fmax_dft < 0.05:
                db.write(image_dft, 
                         ml_ene=image_ml.get_potential_energy(),
                         ml_fmax=np.sqrt(np.sum(image_ml.get_forces()**2, axis=1)).max().item())
                break
            else:
                if train_all:
                    ml_calc.train(image_dft_l, epoches=400, batch_size=1, reload_chk=reload_chk)
                else:
                    ml_calc.train([image_dft], epoches=400, batch_size=1, reload_chk=reload_chk)
                image_ml_retrain = get_atoms_calc([d_atoms.atoms], ml_calc)[0]
                db.write(image_dft, 
                         ml_ene=image_ml.get_potential_energy(),
                         ml_fmax=np.sqrt(np.sum(image_ml.get_forces()**2, axis=1)).max(),
                         ml_ene_retrain=image_ml_retrain.get_potential_energy(),
                         ml_fmax_retrain=np.sqrt(np.sum(image_ml_retrain.get_forces()**2, axis=1)).max().item())
            steps = steps_num
            
            
    end = time.time()
    print("=== Time using: %s seconds for dimer opt!"%(end - start), flush=True)


def run_neb(ini_image, fin_image, ml_calc, vasp_calc, pretrain=False):
    
    start = time.time()
    db = connect("neb.db")
    
    interpolate_num = 3
    converged_num = 0

    images = [ini_image]
    for i in range(interpolate_num):
        image = ini_image.copy()
        image.calc = ml_calc
        images.append(image)
    images.append(fin_image)
    neb = NEB(images, climb=True, allow_shared_calculator=True)
    neb.interpolate(method='idpp', mic=True)

    if pretrain:
        # pretrain
        images_ml = get_atoms_calc(neb.images, ml_calc)
        neb_images_dft = get_atoms_calc(neb.images[1:-1], vasp_calc)
        images_dft = [ini_image] + neb_images_dft + [fin_image]

        ml_calc.train(images_dft, epoches=400, batch_size=len(images_dft))
        images_ml_retrain = get_atoms_calc(neb.images, ml_calc)

        for image_ml, image_dft, image_ml_retrain in zip(images_ml, images_dft, images_ml_retrain):
            db.write(image_dft, 
                     ml_ene=image_ml.get_potential_energy(),
                     ml_fmax=np.sqrt(np.sum(image_ml.get_forces()**2, axis=1)).max(),
                     ml_ene_retrain=image_ml_retrain.get_potential_energy(),
                     ml_fmax_retrain=np.sqrt(np.sum(image_ml_retrain.get_forces()**2, axis=1)).max().item())
    

    while True:
        
        dyn = FIRE(neb, trajectory=None)
        dyn.run(fmax=0.05, steps=50)
        
        images_ml = get_atoms_calc(neb.images, ml_calc)
        neb_images_dft = get_atoms_calc(neb.images[1:-1], vasp_calc)
        images_dft = [ini_image] + neb_images_dft + [fin_image]
        
        ene_arr = [image.get_potential_energy() for image in images_dft]
        fmax_arr = [np.sqrt(np.sum(image.get_forces()**2, axis=1)).max() for image in images_dft]
        top_index = np.argmax(ene_arr)
        
        print("=== is_up_then_down:%s, top_index:%s, fmax_top:%s, ene_top: %s"%(is_up_then_down(ene_arr),
                                                                                top_index,
                                                                                fmax_arr[top_index],
                                                                                ene_arr[top_index]))

        if is_up_then_down(ene_arr) and fmax_arr[top_index] < 0.05:
            converged_num += 1
        
        if converged_num == 3:
            for image_ml, image_dft in zip(images_ml, images_dft):
                db.write(image_dft, 
                         ml_ene=image_ml.get_potential_energy(),
                         ml_fmax=np.sqrt(np.sum(image_ml.get_forces()**2, axis=1)).max().item())
            break
        
        else:
            
            ml_calc.train(images_dft, epoches=400, batch_size=len(images_dft))
            images_ml_retrain = get_atoms_calc(neb.images, ml_calc)
            
            for image_ml, image_dft, image_ml_retrain in zip(images_ml, images_dft, images_ml_retrain):
                db.write(image_dft, 
                        ml_ene=image_ml.get_potential_energy(),
                        ml_fmax=np.sqrt(np.sum(image_ml.get_forces()**2, axis=1)).max(),
                        ml_ene_retrain=image_ml_retrain.get_potential_energy(),
                        ml_fmax_retrain=np.sqrt(np.sum(image_ml_retrain.get_forces()**2, axis=1)).max().item())
            
    ml_calc.trainer.is_debug=False
    ml_calc.trainer.config["cmd"]["checkpoint_dir"] = os.getcwd()
    ml_calc.trainer.save(checkpoint_file="checkpoint.pt", training_state=False)
    ml_calc.trainer.is_debug=True
                  
    end = time.time()
    print("### Time using: %s seconds for neb opt!"%(end - start))


def run_calc(atoms, calc, fix_index=None):
    
    start = time.time()

    if fix_index:
        c = FixAtoms(indices=fix_index)
        atoms.set_constraint(c)
    
    atoms.calc = calc
    calc.calculate(atoms)

    end = time.time()
    print("=== Time using: %s seconds for calc!"%(end - start), flush=True)
