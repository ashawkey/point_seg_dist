# CUDA point-to-line/segment distance

A CUDA function to calculate the distance between a set of points and a set of segments (or lines).

### Install
```bash
pip install git+https://github.com/ashawkey/point_seg_dist

# or locally
git clone https://github.com/ashawkey/point_seg_dist
cd point_seg_dist
pip install .
```

### Usage
```python

import torch
from point_seg_dist import point_seg_dist

M = 1024
N = 20
C = 3 # or 2 (which will be padded to 3d)

p = torch.randn(M, C).cuda()
a = torch.randn(N, C).cuda()
b = torch.randn(N, C).cuda()

d = point_seg_dist(p, a, b) # [M, N]
```
