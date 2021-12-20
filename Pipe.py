import numpy as np

class Pipe:
    # all parameters of the pipe
    # only straight horizontal pipe for now, but might be extended later
    def __init__(self, D, eps, length, N):
        self.D = D  #diameter
        self.eps = eps #wall rougness
        self.N = N
        self.x = np.linspace(0, length, N) # horizontal grid with N cells
        self.dx = length/N
