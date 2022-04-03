import numpy as np
class ReLU():
    def __init__(self) -> None:
        pass

    def forward(self, A):
        return np.maximum(0, A)