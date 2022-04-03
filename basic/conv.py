import numpy as np

class Conv():
    def __init__(self, stride, padding):
        self.stride = stride
        self.padding = padding

    def zero_padding(self, A):
        return np.pad(A, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)

    def conv(self, field, W):
        return np.sum(np.multiply(field, W))

    def forward(self, A, W):
        (m, A_c, A_h, A_w) = A.shape
        (W_c, A_c, f, f) = W.shape

        n_h = int((A_h - f + 2 * self.padding) / self.stride) + 1
        n_w = int((A_w - f + 2 * self.padding) / self.stride) + 1
        Z = np.zeros((m, W_c, n_h, n_w))

        A_pad = self.zero_padding(A)
        for i in range(m):
            one_pic_A = A_pad[i]
            for c in range(W_c):
                for h in range(n_h):
                    for w in range(n_w):
                        v_start = h * self.stride
                        v_end = v_start + f
                        h_start = w * self.stride
                        h_end = h_start + f
                        field = one_pic_A[:, v_start:v_end, h_start:h_end]
                        Z[i, c, h, w] = self.conv(field, W[c, :, :, :])
        return Z
