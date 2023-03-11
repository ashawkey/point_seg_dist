# pragma once

#include <stdint.h>
#include <torch/torch.h>

void point_seg_dist_forward(at::Tensor P, at::Tensor A, at::Tensor B, const uint32_t M, const uint32_t N, const uint32_t C, const bool seg, at::Tensor D);