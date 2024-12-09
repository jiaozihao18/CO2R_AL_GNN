import numpy as np
import sys
from oal_cat.mlCalculator import mlCalculator
from oal_cat.oal_utils import get_atoms_calc
import ase.io
import os
from ocpmodels.common.utils import setup_logging
import random

setup_logging()

def split_train_validation(data, seed, train_ratio):
    random.seed(seed)
    shuffled_data = data[:]
    random.shuffle(shuffled_data)
    train_size = int(len(shuffled_data) * train_ratio)
    train_set = shuffled_data[:train_size]
    validation_set = shuffled_data[train_size:]
    return train_set, validation_set

atoms_l1 = ase.io.read("1cu_h2o/OUTCAR", index=':') # 500 steps, 0.5 ps
atoms_l2 = ase.io.read("1cu_h2o/conti/OUTCAR", index=':') # 4500 steps, 4.5 ps
atoms_l3 = ase.io.read("1cu_h2o/conti2/OUTCAR", index=':') # 5000 steps, 2.5 ps
atoms_l4 = ase.io.read("1cu_h2o/conti3/OUTCAR", index=':') # 15000 steps, 7.5 ps
atoms_l = atoms_l1 + atoms_l2 + atoms_l3 + atoms_l4

# train_l, val_l = split_train_validation(atoms_l, seed=0, train_ratio=0.8)
random.seed(0)
random.shuffle(atoms_l)

unfreeze_blocks_f = ["out_blocks.3.seq_forces.",
                     "out_blocks.3.scale_rbf_F.",
                     "out_blocks.3.dense_rbf_F.",
                     "out_blocks.3.out_forces.",
                     "out_blocks.2.seq_forces.",
                     "out_blocks.2.scale_rbf_F.",
                     "out_blocks.2.dense_rbf_F.",
                     "out_blocks.2.out_forces.",
                     "out_blocks.1.seq_forces.",
                     "out_blocks.1.scale_rbf_F.",
                     "out_blocks.1.dense_rbf_F.",
                     "out_blocks.1.out_forces.",]

unfreeze_blocks_e = ["out_blocks.3.layers.",
                     "out_blocks.3.scale_sum.",
                     "out_blocks.3.dense_rbf.",
                     "out_blocks.3.out_energy.",
                     "out_blocks.2.layers.",
                     "out_blocks.2.scale_sum.",
                     "out_blocks.2.dense_rbf.",
                     "out_blocks.2.out_energy.",
                     "out_blocks.1.layers.",
                     "out_blocks.1.scale_sum.",
                     "out_blocks.1.dense_rbf.",
                     "out_blocks.1.out_energy.",]

# train_forces
checkpoint_path = "gemnet_t_direct_h512_all.pt"
ml_calc = mlCalculator(checkpoint_path=checkpoint_path, cpu=False, unfreeze_blocks=unfreeze_blocks_f)
ml_calc.trainer.config["cmd"]["print_every"] = 20
ml_calc.trainer.config["optim"]["energy_coefficient"] = 0
ml_calc.trainer.config["optim"]["force_coefficient"] = 100
# ml_calc.train(train_l, val_l, epoches=80, batch_size=20)
ml_calc.train(atoms_l, epoches=80, batch_size=20)
ml_calc.trainer.is_debug=False
ml_calc.trainer.config["cmd"]["checkpoint_dir"] = os.getcwd()
ml_calc.trainer.save(checkpoint_file="chk_out/checkpoint_cuh2o_shuffle_f.pt", training_state=False)
ml_calc.trainer.is_debug=True

# train_energy
checkpoint_path = "chk_out/checkpoint_cuh2o_shuffle_f.pt"
ml_calc = mlCalculator(checkpoint_path=checkpoint_path, cpu=False, unfreeze_blocks=unfreeze_blocks_e)
ml_calc.trainer.config["cmd"]["print_every"] = 20
ml_calc.trainer.config["optim"]["energy_coefficient"] = 10
ml_calc.trainer.config["optim"]["force_coefficient"] = 0
# ml_calc.train(train_l, val_l, epoches=80, batch_size=20)
ml_calc.train(atoms_l, epoches=80, batch_size=20)
ml_calc.trainer.is_debug=False
ml_calc.trainer.config["cmd"]["checkpoint_dir"] = os.getcwd()
ml_calc.trainer.save(checkpoint_file="chk_out/checkpoint_cuh2o_shuffle_fe_new.pt", training_state=False)
ml_calc.trainer.is_debug=True