from tinygrad.tensor import Tensor
import tinygrad.nn as nn

class MNISTClassifier:
    def __init__(self):
        self.l1 = nn.Linear(28 * 28, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x)
