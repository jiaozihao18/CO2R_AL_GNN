from ocpmodels.trainers import ForcesTrainer
from ase.db import connect
import yaml
import os
from ocpmodels.common import logger
from ocpmodels.preprocessing import AtomsToGraphs
from finetuna.finetuner_utils.utils import GraphsListDataset, GenericDB
from torch.utils.data import random_split
import torch
from ocpmodels.common.utils import setup_logging, load_config
setup_logging()

def gen_train_val_test(dataset, train_ratio, val_ratio, seed=0):
    
    train_len = int(len(dataset)*train_ratio)
    val_len = int(len(dataset)*val_ratio)
    test_len = len(dataset)-train_len-val_len
    train_subset, val_subset, test_subset = random_split(dataset=dataset, 
                                                         lengths=[train_len, val_len, test_len],
                                                         generator=torch.Generator().manual_seed(seed))
    return train_subset, val_subset, test_subset

def get_data_loader(trainer, atoms_l):
    
    a2g = AtomsToGraphs(max_neigh=50,
                        radius=6,
                        r_energy=True,
                        r_forces=True,
                        r_distances=True,
                        r_edges=False)

    graphs_list = [a2g.convert(atoms, True) for atoms in atoms_l]
    for graph in graphs_list:
        graph.fid = 0
        graph.sid = 0

    graphs_list_dataset = GraphsListDataset(graphs_list)
    data_sampler = trainer.get_sampler(graphs_list_dataset, 
                                       trainer.config["optim"]["batch_size"], shuffle=False)
    data_loader = trainer.get_dataloader(graphs_list_dataset, data_sampler)
    
    return data_sampler, data_loader

timestamp_id = '001'
config_path = "configs/s2ef/all/gemnet/gemnet-dT.yml"
directory_path = 'data/xyz_w'

# config = yaml.safe_load(open(config_path, "r"))
# if "includes" in config:
#     for include in config["includes"]:
#         # Change the path based on absolute path of config_yml
#         path = os.path.join(config_path.split("configs")[0], include)
#         include_config = yaml.safe_load(open(path, "r"))
#         config.update(include_config)

config, _, _ = load_config(config_path)

file_paths = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        file_paths.append(os.path.join(root, file))
atoms_l  = []
for file in file_paths:
    db = connect(file)
    for row in db.select():
        atoms_l.append(row.toatoms())

train_subset, val_subset, test_subset = gen_train_val_test(atoms_l, 0.8, 0.2, seed=0)


config['model']['otf_graph'] = True
config['optim']['force_coefficient'] = 30
config['optim']['energy_coefficient'] = 1
config['optim']['max_epochs'] = 50

trainer = ForcesTrainer(task=config['task'],
                        model=config['model'], 
                        dataset=None,
                        optimizer=config['optim'],
                        normalizer=config['dataset'][0],
                        identifier="s2ef-task",
                        run_dir='./',
                        timestamp_id=timestamp_id,
                        print_every=10,
                        seed=0, 
                        logger="tensorboard",
                        amp=False,
                        is_debug=False)


trainer.train_dataset = GenericDB()
trainer.train_sampler, trainer.train_loader = get_data_loader(trainer, train_subset)

trainer.val_dataset = GenericDB()
trainer.val_sampler, trainer.val_loader = get_data_loader(trainer, val_subset)

trainer.train()