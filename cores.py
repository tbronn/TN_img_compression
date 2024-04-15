import utils
import torch
import tensornetwork as tn


def init_cores(num_cores, core_base, seed=1, bond_dim=1):
    list_cores = []

    utils.set_random_generator(seed)
    for i in range(num_cores):
        list_indices = [bond_dim + (core_base - 1) * (i == j)
                        for j in range(num_cores)]
        
        core = torch.rand(list_indices)
        
        core = tn.Node(core)

        list_cores.append(core)

    return list_cores


def init_bonds(list_cores):
    num_cores = len(list_cores)
    for i in range(num_cores):
        Gi = list_cores[i]
        for j in range(i + 1, num_cores):
            Gj = list_cores[j]

            tn.connect(Gi[j], Gj[i])
        
    return


def reconstruct_tensor(list_cores):
    recon_tn = list_cores[0]
    for core in list_cores[1:]:
        recon_tn = tn.contract_between(recon_tn, core)

    init_bonds(list_cores)

    return recon_tn.tensor


def add_slice(core, axis):
    new_slice_shape = [axis_dim if j!=axis else 1
                        for j, axis_dim in enumerate(core.shape)]
    new_slice = torch.randn(new_slice_shape, device=core.device)
    new_core = torch.cat((core, new_slice), axis)
    return new_core
