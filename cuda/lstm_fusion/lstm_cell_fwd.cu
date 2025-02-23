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
    scalar_t* __restrict__ i_gates, 
    scalar_t* __restrict__ h_gates, 
    scalar_t* __restrict__ i_b, 
    scalar_t* __restrict__ h_b, 
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
  
  const int64_t col1 = col + state_size;
  const int64_t col2 = col1 + state_size;
  const int64_t col3 = col2 + state_size;
  i_gates[id0] = i_gates[id0] + h_gates[id0] + i_b[col] + h_b[col];
  i_gates[id1] = i_gates[id1] + h_gates[id1] + i_b[col1] + h_b[col1];
  i_gates[id2] = i_gates[id2] + h_gates[id2] + i_b[col2] + h_b[col2];
  i_gates[id3] = i_gates[id3] + h_gates[id3] + i_b[col3] + h_b[col3];
  i_gates[id0] = sigmoid(i_gates[id0]); // i_gate
  i_gates[id1] = sigmoid(i_gates[id1]); // f_gate
  i_gates[id2] = tanh(i_gates[id2]); // c_gate
  i_gates[id3] = sigmoid(i_gates[id3]); // o_gate
  
  const int64_t idc = row * state_size + col;
  c_new[idc] = i_gates[id1] * c_prev[idc] + i_gates[id0] * i_gates[id2];
  h_new[idc] = i_gates[id3] * tanh(c_new[idc]);
}

vector<at::Tensor> lstm_cell_act_forward_cuda(
    torch::Tensor &input, 
    torch::Tensor &h_prev,
    torch::Tensor &c_prev,
    torch::Tensor &wi, 
    torch::Tensor &wh,
    torch::Tensor &bi,
    torch::Tensor &bh)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threads = prop.maxThreadsPerBlock;
    const dim3 blocks((c_prev.numel() + threads - 1) / threads);
    torch::Tensor c_new = torch::zeros_like(c_prev);
    torch::Tensor h_new = torch::zeros_like(c_prev);
    torch::Tensor i_gates = at::matmul(input, wi.t());
    torch::Tensor h_gates = at::matmul(h_prev, wh.t());
    // i_gates += h_gates + bi + bh;

    
    // cout << "shapes:  " << i_gates.sizes() << " " << h_gates.sizes() << " " << bi.sizes() << " " << bh.sizes() << c_prev.sizes() << "\n";
    AT_DISPATCH_FLOATING_TYPES(i_gates.type(), "lstm_cell_act_forward", ([&] {
        lstm_cell_act_fwd<scalar_t><<<blocks, threads>>>(
            i_gates.data<scalar_t>(), \
            h_gates.data<scalar_t>(), \
            bi.data<scalar_t>(), \
            bh.data<scalar_t>(), \
            c_prev.data<scalar_t>(), \
            c_new.data<scalar_t>(), \
            h_new.data<scalar_t>(), \
            c_prev.numel(), \
            c_prev.size(-1));
    }));
        
    return {h_new, c_new};
}