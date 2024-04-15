import utils
import cores
from module import GreedyTensorNetwork
import copy
import torch
import numpy as np

class Trainer(utils.HyperParameters):
    """The base class for training Greedy-TN model with data"""
    def __init__(self, max_steps=10, max_epochs_cores=30, verbose=False, precision=1e-6, loss_val_bound=0.2, num_gpus=0):
        self.save_hyperparameters()
        if num_gpus > 0 and utils.gpu_is_available():
            print(f"Run with GPU support.\n")
        else:
            print(f"Run without GPU support.\n")
        self.gpus = [utils.set_gpu(i) for i in range(min(self.num_gpus, utils.get_num_gpus()))]
        self.dict_res = {}
        self.list_loss = []
        self.list_num_params = []

    def prepare_target_tensor(self, target_tensor):
        if self.gpus:
            target_tensor.use_gpus(self.gpus[0])
        self.target_tensor = target_tensor.get_tensor()

    def prepare_model(self, model, tmp_model_flag=False):
        if self.gpus:
            model.use_gpus(self.gpus[0])
        if not tmp_model_flag:
            self.model = model

    def find_best_bond(self):
        loss_val = np.inf
        self.add_bond_flag = False
        self.new_bond_ind = (-1, -1)
        for i in range(self.model.num_cores):
            for j in range(i + 1, self.model.num_cores):

                tmp_list_cores = copy.deepcopy(self.model.list_cores)
                
                tmp_model = GreedyTensorNetwork(list_cores=tmp_list_cores)

                self.prepare_model(tmp_model, tmp_model_flag=True)
                
                bond = (i, j)
                
                self.increment_rank(tmp_model, bond)
                
                tmp_loss_val = tmp_model.loss(self.target_tensor, cores.reconstruct_tensor(tmp_model.list_cores))

                if tmp_loss_val < loss_val:
                    loss_val = tmp_loss_val
                    self.new_bond_ind = (i, j)

        if self.new_bond_ind[0] != self.new_bond_ind[1]:
            self.add_bond_flag = True

        return

    def increment_rank(self, model, bond):
        i, j = bond

        model.list_cores[i].tensor = cores.add_slice(model.list_cores[i].tensor, j)
        model.list_cores[j].tensor = cores.add_slice(model.list_cores[j].tensor, i)

        self.fit_cores(model, cores_indices=[i, j])
        
    def fit_cores(self, model, cores_indices=None):
        optim = model.configure_optimizers(self.target_tensor) 
    
        self.epoch = 0
        prev_loss_val = np.inf
        
        for self.epoch in range(self.max_epochs_cores):

            optim.step(cores_indices)
            
            loss_val = model.loss(self.target_tensor, cores.reconstruct_tensor(model.list_cores))
            
            if torch.abs(loss_val - prev_loss_val) < self.precision:
                #if self.verbose:
                #    print(f"ALS converged at {self.epoch} iteration")
                break
            prev_loss_val = loss_val

    def fit(self, model, target_tensor):
        self.prepare_target_tensor(target_tensor)
        self.prepare_model(model)
        self.optim = model.configure_optimizers(self.target_tensor)

        self.fit_cores(self.model)

        for step in range(self.max_steps):

            self.find_best_bond()
            
            if self.add_bond_flag: 
                self.increment_rank(self.model, self.new_bond_ind)
            else:
                print("No new bond reduces the error\n")
                break
                
            loss_val = self.model.loss(self.target_tensor, cores.reconstruct_tensor(self.model.list_cores))
            num_params = self.model.get_num_params()
            
            if loss_val < self.loss_val_bound:
                self.dict_res[f'step={step}, loss_val={loss_val}, num_params={num_params}'] = self.model.get_matrix()
                
            if self.verbose:
                print('-'*5 + f" step={step} " + '-'*5)
                if self.gpus:
                    loss_val = loss_val.cpu()
                self.list_loss.append(loss_val)
                self.list_num_params.append(num_params)
                print(f"loss: {self.list_loss[-1]}")
                print(f"params: {self.list_num_params[-1]}")
