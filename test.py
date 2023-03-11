import numpy as np
import torch
import time
from point_seg_dist import point_seg_dist

def lineseg_dists(p, a, b):
    # Handle case where p is a single point, i.e. 1d array.
    p = np.atleast_2d(p)

    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)
    if len(c.shape) > 1:
        c = np.linalg.norm(c, axis=-1)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)


# numpy
C = 3
p = np.random.randn(10240, C)
a = np.random.randn(20, C)
b = np.random.randn(20, C)

_t0 = time.time()
ds = []
for i in range(a.shape[0]):
    d = lineseg_dists(p, a[i], b[i])
    ds.append(d)
ds = np.stack(ds, axis=1)
_t1 = time.time()
print(f'[NUMPY] {_t1 - _t0:.6f}')

# torch
pp = torch.from_numpy(p).to('cuda')
aa = torch.from_numpy(a).to('cuda')
bb = torch.from_numpy(b).to('cuda')

_t0 = time.time()
dds = point_seg_dist(pp, aa, bb).cpu().numpy()
torch.cuda.synchronize()
_t1 = time.time()
print(f'[CUDA] {_t1 - _t0:.6f}')

# print(ds[:3, :3])
# print(dds[:3, :3])

print(np.allclose(ds, dds, atol=1e-6))
