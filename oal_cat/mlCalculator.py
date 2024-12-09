import ase.io
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
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import time
# from typing import Optional
import numpy as np

# setup_logging() # default logging level is "WARNING", ingoring "INFO" level formation

class mlCalculator(OCPCalculator):
    def __init__(
        self, 
        config_yml: str | None = None, 
        checkpoint_path: str | None = None, 
        trainer: str | None = None,
        cutoff: int = 6, 
        max_neighbors: int = 50, 
        cpu: bool = True,
        unfreeze_blocks = [],
        energy_coefficient: int = 0) -> None:
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        config = checkpoint["config"]
        
        config["optim"]['num_workers'] = 0
        config["optim"]['lr_initial'] = 0.0003
        config["optim"]['factor'] = 0.9
        config["optim"]['patience'] = 3
        config["optim"]['checkpoint_every'] = 100000
        # config["optim"]['scheduler_loss'] = "train"
        config["optim"]["eval_every"] = 1 # scheduler "ReduceLROnPlateau" work after each evaluation
        
        config["optim"]["optimizer_params"] = {"weight_decay": 0,"eps": 1e-8}
        
        config["optim"]["energy_coefficient"] = energy_coefficient
        
        config["normalizer"] = {'normalize_labels': True,
                                'target_mean': -0.7554450631141663,
                                'target_std': 2.887317180633545,
                                'grad_target_mean': 0.0,
                                'grad_target_std': 2.887317180633545}
        
        super().__init__(config, checkpoint_path, trainer, cutoff, max_neighbors, cpu)
        
        if unfreeze_blocks == []:
            unfreeze_blocks = ["out_blocks.3.seq_forces.",
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
        self.unfreeze_blocks = unfreeze_blocks
        
        self.freeze_blocks()
        
        self.model_name = self.trainer.config['model']
        self.chk_path = checkpoint_path
        # self.trainer.config['cmd']['print_every'] = 100000
        
    def freeze_blocks(self):
        # first freeze all weights within the loaded model
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
        # then unfreeze certain weights within the loaded model
        for name, param in self.trainer.model.named_parameters():
            for block_name in self.unfreeze_blocks:
                if block_name in name:
                    param.requires_grad = True
        
    def get_data_loader(self, atoms_l):
    
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
        data_sampler = self.trainer.get_sampler(graphs_list_dataset, 
                                                self.trainer.config["optim"]["batch_size"], shuffle=False)
        data_loader = self.trainer.get_dataloader(graphs_list_dataset, data_sampler)
        
        return data_sampler, data_loader
    
    def train(self, atoms_l, epoches=400, batch_size=1, reload_chk=False):
        
        if reload_chk:
            self.trainer.load_model()
            self.freeze_blocks()
            self.trainer.load_checkpoint(self.chk_path)
        
        """Clear all information from old calculation."""
        self.atoms = None
        self.results = {}
        
        self.trainer.step = 0
        self.trainer.epoch = 0
        
        self.trainer.load_optimizer()
        self.trainer.load_extras()
        
        self.trainer.config['optim']['batch_size'] = batch_size
        self.trainer.config['optim']['max_epochs'] = epoches
        
        self.trainer.train_dataset = GenericDB()
        self.trainer.train_sampler, self.trainer.train_loader = self.get_data_loader(atoms_l)
        
        # val the same as train
        self.trainer.val_dataset = GenericDB()
        self.trainer.val_sampler, self.trainer.val_loader = self.get_data_loader(atoms_l)
        
        start = time.time()
        self.trainer.train(disable_eval_tqdm=True)
        end = time.time()
        
        print("Time to train %s on %s pts: %s seconds" 
              %(str(self.model_name), len(atoms_l), str(end - start)))

