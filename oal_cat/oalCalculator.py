from ase.calculators.calculator import Calculator
from ase import Atoms
from .oal_utils import get_atoms_calc, get_error
import numpy as np
import time
import ase.db
import random
from numpy import ndarray

class oalCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        ml_calc,
        parent_calc,
        initial_points: int = 5,
        query_every_n_steps: int = 50,
        fmax_verify_threshold: float = 0.05,
        partial_fit_num: int = 1,
        train_epochs: int = 400,
        oal_db_name: str = None) -> None:
        
        Calculator.__init__(self)
        
        self.ml_calc = ml_calc
        self.parent_calc = parent_calc
        self.parent_dataset = []
        self.initial_points = initial_points
        self.query_every_n_steps = query_every_n_steps
        self.fmax_verify_threshold = fmax_verify_threshold
        
        self.partial_fit_num = partial_fit_num
        
        self.train_epochs = train_epochs
        
        self.parent_calls = 0
        self.curr_step = 0
        self.steps_since_last_query = 0
        
        print("Parent calc is :", self.parent_calc)
        self.parent_calc_pausable = False
        if hasattr(self.parent_calc, "pause"):
            self.parent_calc_pausable = True
            self.parent_calc._pause_calc()
        
        self.oal_db_name = oal_db_name

    
    def init_info(self):
        self.info = {
            "check": None,
            "ml_energy": None,
            "ml_forces": None,
            "ml_fmax": None,
            "parent_energy": None,
            "parent_forces": None,
            "parent_fmax": None,
            "retrained_energy": None,
            "retrained_forces": None,
            "retrained_fmax": None,
            "energy_error": None,
            "forces_error": None,
            "relative_forces_error": None,
            "retrained_energy_error": None,
            "retrained_forces_error": None,
            "retrained_relative_forces_error": None,
            "training_time": None,
            "parent_time": None,
        }
        
    def write_db(self, atoms):
        # write to ASE db
        if self.oal_db_name is not None:
            random.seed(time.time())
            dict_to_write = {}
            for key, value in self.info.items():
                dict_to_write[key] = value
                if value is None:
                    dict_to_write[key] = "-"
                elif type(value) is ndarray:
                    dict_to_write[key] = str(value)
                    
            with ase.db.connect(self.oal_db_name) as asedb:
                asedb.write(atoms, key_value_pairs=dict_to_write)
    
    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        
        self.curr_step += 1
        self.steps_since_last_query += 1

        energy, forces = self.get_energy_and_forces(atoms)
        
        self.write_db(atoms)
        
        self.results["energy"] = energy
        self.results["forces"] = forces
    
    def get_energy_and_forces(self, atoms):
        
        self.init_info()
        
        if len(self.parent_dataset) < self.initial_points:
            energy, forces = self.add_data_and_retrain(atoms)
            
            return energy, forces
        
        else:
            if self.steps_since_last_query % self.query_every_n_steps == 0:
                print("%s steps since last query, querying every %s steps: check with parent"
                      %(self.steps_since_last_query, self.query_every_n_steps))
                
                energy, forces = self.add_data_and_retrain(atoms)
            else:
                atoms_ml = get_atoms_calc([atoms], self.ml_calc)[0]
                
                energy_ml = atoms_ml.get_potential_energy(apply_constraint=False)
                forces_ml = atoms_ml.get_forces(apply_constraint=False)
                fmax = np.sqrt((atoms_ml.get_forces(apply_constraint=True)**2).sum(axis=1).max())
                
                if fmax < self.fmax_verify_threshold:
                    print("Force below threshold: check with parent")
                    energy, forces = self.add_data_and_retrain(atoms)
                else:
                    self.info["check"] = False
                    self.info["ml_energy"] = energy_ml
                    self.info["ml_forces"] = forces_ml
                    self.info["ml_fmax"] = fmax
                    
                    energy, forces = energy_ml, forces_ml
                
            return energy, forces
                
    def add_data_and_retrain(self, atoms):
        
        self.info["check"] = True
        
        self.parent_calls += 1
        self.steps_since_last_query = 0
        
        start = time.time()
        if self.parent_calc_pausable:
            self.parent_calc._resume_calc()
        atoms_p = get_atoms_calc([atoms], self.parent_calc)[0]
        if self.parent_calc_pausable:
            self.parent_calc._pause_calc()
        end = time.time()
        
        parent_time = end - start
        print("Time to call parent (call #%s): %s"%(self.parent_calls, parent_time))
        
        self.parent_dataset.append(atoms_p)
        
        atoms_ml = get_atoms_calc([atoms], self.ml_calc)[0]
        
        start = time.time()
        if len(self.parent_dataset) == self.initial_points:
            self.ml_calc.train(self.parent_dataset, epoches=self.train_epochs, batch_size=1)
            
        if len(self.parent_dataset) > self.initial_points:
            self.ml_calc.train(self.parent_dataset[-self.partial_fit_num:], epoches=self.train_epochs, batch_size=1)
        end = time.time()
        training_time = start - end
        
        atoms_ml_retrained = get_atoms_calc([atoms], self.ml_calc)[0]
        
        
        self.info["ml_energy"] = atoms_ml.get_potential_energy(apply_constraint=False)
        self.info["ml_forces"] = atoms_ml.get_forces(apply_constraint=False)
        self.info["ml_fmax"] = np.sqrt((atoms_ml.get_forces(apply_constraint=True)**2).sum(axis=1).max())
        
        self.info["parent_energy"] = atoms_p.get_potential_energy(apply_constraint=False)
        self.info["parent_forces"] = atoms_p.get_forces(apply_constraint=False)
        self.info["parent_fmax"] = np.sqrt((atoms_p.get_forces(apply_constraint=True)**2).sum(axis=1).max())
        
        self.info["retrained_energy"] = atoms_ml_retrained.get_potential_energy(apply_constraint=False)
        self.info["retrained_forces"] = atoms_ml_retrained.get_forces(apply_constraint=False)
        self.info["retrained_fmax"] = np.sqrt((atoms_ml_retrained.get_forces(apply_constraint=True)**2).sum(axis=1).max())
        
        error_dict = get_error(atoms_p, atoms_ml)
        self.info["energy_error"] = error_dict["ene_error"]
        self.info["forces_error"] = error_dict["forces_error"]
        self.info["relative_forces_error"] = error_dict["relative_forces_error"]
        
        error_dict = get_error(atoms_p, atoms_ml_retrained)
        self.info["retrained_energy_error"] = error_dict["ene_error"]
        self.info["retrained_forces_error"] = error_dict["forces_error"]
        self.info["retrained_relative_forces_error"] = error_dict["relative_forces_error"]
        
        self.info["training_time"] = training_time
        self.info["parent_time"] = parent_time
        
        energy = atoms_p.get_potential_energy(apply_constraint=False)
        forces = atoms_p.get_forces(apply_constraint=False)
        
        return energy, forces