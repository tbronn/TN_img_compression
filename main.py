from module import GreedyTensorNetwork
from data_module import TargetImage
from trainer import Trainer
import tensornetwork as tn
import time

tn.set_default_backend("pytorch")

def main():
    t0 = time.time()
    target_tensor = TargetImage("cruise_ship.jpg", height=256, width=256)
    model = GreedyTensorNetwork(num_cores=8, core_base=4)
    trainer = Trainer(max_steps=17, verbose=True, num_gpus=1)
    trainer.fit(model, target_tensor)
    t1 = time.time()
    print("Total time:", t1 - t0)


if __name__ == '__main__':
    main()