import utils
import torch
import numpy as np
from tensornetwork import ncon



class ALS(utils.HyperParameters):
    """Alternating Least Squares for tensor network."""
    def __init__(self, list_cores, target_tensor):
        self.save_hyperparameters()
        self.num_cores = len(self.list_cores)


    def get_list_bonds(self, tensor_list):
        self.num_cores = len(tensor_list)
        num_bonds = self.num_cores * (self.num_cores - 1) // 2
        matrix_bonds = torch.triu(torch.ones((self.num_cores, self.num_cores), dtype=torch.long), diagonal=1)
        matrix_bonds[matrix_bonds == 1] = torch.arange(1, num_bonds + 1)
        matrix_bonds = matrix_bonds + matrix_bonds.T
        torch.diagonal(matrix_bonds).copy_(-torch.arange(1, self.num_cores + 1))
        return matrix_bonds.tolist()
    

    def unfold(self, modes):
        if not utils.is_iterable(modes):
            modes = (modes,)
        row_dims = [self.target_tensor.shape[i] for i in modes]
        M = torch.movedim(self.target_tensor, modes, tuple(range(len(modes))))
        return M.reshape([np.prod(row_dims), -1])
            

    def least_squares_solver(self, core_ind):

        list_bonds = self.get_list_bonds(self.torch_list_cores)

        tmp_list_cores = [core for ind, core in enumerate(self.torch_list_cores) if ind != core_ind]
    
        it = -self.num_cores - 1
        tmp_list_bonds = []

        for ind, bonds_ind in enumerate(list_bonds):
            if ind != core_ind:
                tmp_list_bonds.append(bonds_ind)
                tmp_list_bonds[-1][core_ind] = it
                it -= 1

        tn_without_core_ind = ncon(tmp_list_cores, tmp_list_bonds)

        tn_without_core_ind = tn_without_core_ind.reshape([np.prod(
                                                            tn_without_core_ind.shape[:self.num_cores - 1]), 
                                                           np.prod(
                                                            tn_without_core_ind.shape[self.num_cores - 1:])])

        axes = list(range(self.num_cores))
        axes.insert(0, axes.pop(core_ind))

        target_tensor_unfold = self.unfold(core_ind)

        sol_lstsq = torch.linalg.lstsq(tn_without_core_ind, target_tensor_unfold.T, driver='gels').solution.T

        tn_shape = list(self.torch_list_cores[core_ind].shape)
        tn_shape.insert(0, tn_shape.pop(core_ind))
        sol_lstsq = sol_lstsq.reshape(tn_shape)
        sol_lstsq = torch.movedim(sol_lstsq, 0, core_ind)

        return sol_lstsq


    def step(self, cores_indices=None):
        if cores_indices is None:
            cores_indices = range(self.num_cores)
        self.torch_list_cores = [t.tensor for t in self.list_cores]

        for core_ind in cores_indices:
            
            self.torch_list_cores[core_ind] = self.least_squares_solver(core_ind)
            
            self.list_cores[core_ind].tensor = self.torch_list_cores[core_ind]