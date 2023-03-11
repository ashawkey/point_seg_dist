#include <torch/extension.h>

#include "point_seg_dist.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("point_seg_dist_forward", &point_seg_dist_forward, "point_seg_dist forward (CUDA)");
}