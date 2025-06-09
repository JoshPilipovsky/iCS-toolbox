import numpy as np

class NonlinearSDE:
    """
    dx = f(x,u,t) dt + g(x,u) dw
    """
    def __init__(self, dim_x, dim_u, dim_w):
        self.n, self.m, self.d = dim_x, dim_u, dim_w