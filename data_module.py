import utils
from PIL import Image
import math
import torchvision
import tensornetwork as tn
import matplotlib.pyplot as plt



class TargetData(utils.HyperParameters):
    """The base class of data"""
    def __init__(self, root='../data/'):
        self.save_hyperparameters()

    def get_tensor(self):
        raise NotImplementedError
        
    def visualize(self):
        raise NotImplementedError
        

class TargetImage(TargetData):
    """Target tensor from a given grayscale image."""
    def __init__(self, file_name, height=128, width=128, core_base=4, bond_dim=1, order='F'):
        super().__init__()
        self.save_hyperparameters()
        self.load_image(file_name)
        self.matrix = self.crop_image()
        self.target_tensor = self.image_to_tensor()
    
    def load_image(self, file_name):
        try:
            self.image = Image.open(f"{self.root}{file_name}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file '{file_name}' not found in '{self.root}'")
        except OSError as e:
            raise OSError(f"Error opening image file '{file_name}': {e}")
            
    def crop_image(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop((self.height, self.width)),
                                    torchvision.transforms.ToTensor()])
        return transform(self.image)[0, :, :]

    def use_gpus(self, device):
        self.target_tensor.tensor = self.target_tensor.tensor.to(device)
    
    def image_to_tensor(self):
        total_numel = self.matrix.numel()
        num_cores = int(math.log(total_numel, self.core_base))
        target_tensor_shape = [self.core_base] * num_cores
        target_tensor = utils.reshape_fortran(self.matrix, shape=target_tensor_shape)
        target_tensor = tn.Node(target_tensor)
        return target_tensor
    
    def get_tensor(self):
        return self.target_tensor.tensor
        
    def get_matrix(self):
        return self.matrix.numpy()

    def visualize(self, option='Croped'):
        if option == 'Original':
            self.image.show()
        elif option == 'Croped':
            plt.imshow(self.matrix, cmap='gray')
        else:
            raise ValueError("Error: option parameter should be 'Original' or 'Croped'")
        return