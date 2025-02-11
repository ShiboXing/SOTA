#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

using namespace std;

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t &z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t tanh(scalar_t &z) {
  scalar_t exp_res = exp(-2 * z);
  return (1.0 - exp_res) / (1.0 + exp_res);
}

template <typename scalar_t>
__global__ void lstm_cell_act_fwd(
    scalar_t* __restrict__ gates, 
    scalar_t* __restrict__ c_prev, 
    scalar_t* __restrict__ c_new, 
    scalar_t* __restrict__ h_new, 
    const int64_t c_prev_numel, 
    const int64_t state_size) {
  
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_prev_numel) return;
  const int64_t col = i % state_size;
  const int64_t row = i / state_size;
  const int64_t base = row * state_size * 4;
  const int64_t tanh_st = 2 * state_size, tanh_ed = tanh_st + state_size;
  const int64_t id0 = base + col;
  const int64_t id1 = id0 + state_size;
  const int64_t id2 = id1 + state_size;
  const int64_t id3 = id2 + state_size;
  gates[id0] = sigmoid(gates[id0]); // i_gate
  gates[id1] = sigmoid(gates[id1]); // f_gate
  gates[id2] = tanh(gates[id2]); // c_gate
  gates[id3] = sigmoid(gates[id3]); // o_gate
  
  const int64_t idc = row * state_size + col;
  c_new[idc] = gates[id1] * c_prev[idc] + gates[id0] * gates[id2];
  h_new[idc] = gates[id3] * tanh(c_new[idc]);
}

vector<at::Tensor> lstm_cell_act_forward_cuda(
    torch::Tensor &gates,
    torch::Tensor &c_prev)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threads = prop.maxThreadsPerBlock;
    const dim3 blocks((c_prev.numel() + threads - 1) / threads);
    torch::Tensor c_new = torch::zeros_like(c_prev);
    torch::Tensor h_new = torch::zeros_like(c_prev);

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lstm_cell_act_forward", ([&] {
        lstm_cell_act_fwd<scalar_t><<<blocks, threads>>>(
            gates.data<scalar_t>(), \
            c_prev.data<scalar_t>(), \
            c_new.data<scalar_t>(), \
            h_new.data<scalar_t>(), \
            c_prev.numel(), \
            c_prev.size(-1));
    }));
    
    vector<torch::Tensor> chunks = torch::chunk(gates, 4, 2);
    torch::Tensor i_gate = chunks[0], 
        f_gate = chunks[1],
        c_gate = chunks[2],
        o_gate = chunks[3];
        
    return {h_new, c_new};
}