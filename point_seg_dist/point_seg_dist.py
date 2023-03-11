import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

import _point_seg_dist as _backend

class point_seg_dist_func(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, P, A, B, seg=True):
        # P: [M, C]
        # A: [N, C]
        # B: [N, C]
        # seg: treat AB as segment, or line.

        assert P.shape[1] == A.shape[1] and P.shape[1] == B.shape[1]
        assert A.shape[0] == B.shape[0]

        M, C = P.shape
        N = A.shape[0]
        
        D = torch.empty(M, N, dtype=A.dtype, device=A.device)

        _backend.point_seg_dist_forward(P, A, B, M, N, C, seg, D)

        return D

def point_seg_dist(P, A, B, seg=True):
    return point_seg_dist_func.apply(P, A, B, seg)