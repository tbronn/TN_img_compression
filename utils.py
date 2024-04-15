from collections.abc import Iterable
import inspect
import torch


def set_gpu(i=0):
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

def get_num_gpus():
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

def gpu_is_available():
    return torch.cuda.is_available()

def is_iterable(obj):
    return isinstance(obj, Iterable)

def set_random_generator(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return

def reshape_fortran(Tensor, shape):
    if len(Tensor.shape) > 0:
        Tensor = Tensor.permute(*reversed(range(len(Tensor.shape))))
    return Tensor.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

    