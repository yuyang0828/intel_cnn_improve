import numpy as np

class BN():
    def __init__(self, gamma, beta, eps) -> None:
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    def forward(self, A):
        N, C, H, W = A.shape
        A = np.transpose(A,(1, 0, 2, 3))
        
        A = A.reshape(C, -1)
        sample_mean = np.mean(A, axis = 1).reshape(C, 1)
        sample_var = np.var(A, axis= 1).reshape(C, 1)
        std = np.sqrt(sample_var + self.eps)
        c, n = A.shape
        xn = np.zeros((c, n))
        xn = (A - sample_mean) / std

        # param gamma and beta will be learned in BP
        out = self.gamma * xn + self.beta

        out = out.reshape(C, N, H, W)
        out = np.transpose(out,(1, 0, 2, 3))
        return out