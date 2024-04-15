import utils
import cores
import opt
import math
import torch
import numpy as np
import matplotlib.pyplot as plt




class GreedyTensorNetwork(utils.HyperParameters):
    """The Greedy Tensor Network model."""
    ### TO DO: init factors with normal factors (add arguments sigma)
    def __init__(self, list_cores=None, num_cores=None, core_base=4, seed=1):
        # init_cores must be tn instance
        self.save_hyperparameters()
        self.dev = None
        
        if self.num_cores and not self.list_cores:
            self.setup_network()
        elif self.list_cores and not self.num_cores:
            self.num_cores = len(self.list_cores)
        else:
            assert self.num_cores is not None and self.list_cores is not None
            assert self.num_cores == len(self.list_cores) and self.num_cores
       
    def setup_network(self):
        self.list_cores = cores.init_cores(self.num_cores, self.core_base, self.seed)
        cores.init_bonds(self.list_cores)
        print(f"Shape of the initial tensor \n {self.get_tensor().shape}")

    def use_gpus(self, device):
        self.dev = device
        for core in self.list_cores:
            core.tensor = core.tensor.to(device)
    
    def get_tensor(self):
        return cores.reconstruct_tensor(self.list_cores)
    
    def loss(self, target_tensor, pred_tensor, normalized=True):
        if normalized:
            return torch.norm(target_tensor.squeeze() - pred_tensor.squeeze()) / torch.norm(target_tensor)
        else:
            return torch.norm(target_tensor.squeeze() - pred_tensor.squeeze())
    
    def get_num_params(self, averaged=True):
        torch_list_cores = [t.tensor for t in self.list_cores]
        self.num_params = np.sum([core.numel() for core in torch_list_cores])
        if averaged:
            total_numel = self.get_tensor().numel()
            self.num_params = self.num_params / total_numel
            return self.num_params
        else:
            return self.num_params

    def configure_optimizers(self, target_tensor):
        return opt.ALS(self.list_cores, target_tensor)
        
    def visualize(self):
        img = self.get_matrix()
        plt.imshow(img, cmap='gray')
        return
    
    def get_matrix(self):
        d = int(math.sqrt(self.get_tensor().numel()))
        matrix = utils.reshape_fortran(self.get_tensor(), shape=(d, d))
        if self.dev:
            matrix = matrix.cpu()
        return matrix