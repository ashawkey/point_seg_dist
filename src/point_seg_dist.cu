#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdexcept>

#include "helper_math.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}


// P: [M, C]
// A: [N, C]
// B: [N, C]
// D: [M, N]
template <typename scalar_t, uint32_t C>
__global__ void kernel_point_seg_dist_forward(
    const scalar_t * __restrict__ P,
    const scalar_t * __restrict__ A,
    const scalar_t * __restrict__ B,
    const uint32_t M, const uint32_t N, const bool seg,
    scalar_t * D
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= M * N) return;

    const uint32_t m = i / N;
    const uint32_t n = i % N;

    // locate
    P += m * C;
    A += n * C;
    B += n * C;
    D += i;

    float3 a, b, p;
    if (C == 2) {
        p = make_float3(P[0], P[1], 0);
        a = make_float3(A[0], A[1], 0);
        b = make_float3(B[0], B[1], 0);
    } else {
        p = make_float3(P[0], P[1], P[2]);
        a = make_float3(A[0], A[1], A[2]);
        b = make_float3(B[0], B[1], B[2]);
    }

    // point-to-seg distance
    if (a.x == b.x && a.y == b.y && a.z == b.z) {
        D[0] = length(p - a);
        return;
    }

    float3 d = normalize(b - a);
    float h = 0;

    if (seg) {
        float s = dot(a - p, d);
        float t = dot(p - b, d);
        h = fmaxf(0, fmaxf(s, t));
    }

    float c = length(cross(p - a, d));
    
    D[0] = sqrtf(h * h + c * c);
}



void point_seg_dist_forward(at::Tensor P, at::Tensor A, at::Tensor B, const uint32_t M, const uint32_t N, const uint32_t C, const bool seg, at::Tensor D) {
    CHECK_CUDA(P);
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    
    CHECK_CONTIGUOUS(P);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);

    CHECK_IS_FLOATING(P);
    CHECK_IS_FLOATING(A);
    CHECK_IS_FLOATING(B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    A.scalar_type(), "point_seg_dist_forward", ([&] {
        static constexpr uint32_t N_THREADS = 256;
        switch (C) {
            case 2: kernel_point_seg_dist_forward<scalar_t, 2><<<div_round_up(M * N, N_THREADS), N_THREADS>>>(P.data_ptr<scalar_t>(), A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, N, seg, D.data_ptr<scalar_t>()); break;
            case 3: kernel_point_seg_dist_forward<scalar_t, 3><<<div_round_up(M * N, N_THREADS), N_THREADS>>>(P.data_ptr<scalar_t>(), A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, N, seg, D.data_ptr<scalar_t>()); break;
            default: throw std::runtime_error{"ponit_seg_dist: C must be 3."};
        }
    }));
}