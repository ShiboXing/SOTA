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
    const int64_t c_prev_numel, 
    const int64_t state_size) {
  
  // const int index = blockIdx.y * state_size + column;
  // const int gates_row = blockIdx.y * (state_size * 3);
  // if (threadIdx.x == 700 && threadIdx.y == 0 && threadIdx.z == 0 && 
  //   blockIdx.x == 10 && blockIdx.y == 0 && blockIdx.z == 0) {
  //       printf("blockDim.x: %d, blockIdx.x: %d, blockDim.y: %d blockIdx.y: %d\n", blockDim.x, blockIdx.x, blockDim.y, blockIdx.y);
  //       printf("threadIdx.x: %d, threadIdx.y: %d\n", threadIdx.x, threadIdx.y);
  //       printf("gridDim.x: %d, gridDim.y: %d\n", gridDim.x, gridDim.y);
  //       printf("col, index, gates_rows: %d", column);
  //       printf("state_size: %d", state_size);
  //   }
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
  gates[id0] = sigmoid(gates[id0]);
  gates[id1] = sigmoid(gates[id1]);
  gates[id2] = tanh(gates[id2]);
  gates[id3] = sigmoid(gates[id3]);
}

vector<at::Tensor> lstm_cell_act_forward_cuda(
    torch::Tensor &gates,
    torch::Tensor &c_prev)
{
    // cout << "i_gate: " << i_gate << "\n";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threads = prop.maxThreadsPerBlock;
    const dim3 blocks((c_prev.numel() + threads - 1) / threads);
    // cout << "threads per block called:" << threads << "\n";
    // std::cout << "Blocks called: (" << blocks.x << ", " << blocks.y << ", " << blocks.z << ")" << std::endl;

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lstm_cell_act_forward", ([&] {
        lstm_cell_act_fwd<scalar_t><<<blocks, threads>>>(
            gates.data<scalar_t>(), \
            c_prev.data<scalar_t>(), \
            c_prev.numel(), \
            c_prev.size(-1));
    }));
    
    vector<torch::Tensor> chunks = torch::chunk(gates, 4, 2);
    torch::Tensor i_gate = chunks[0], 
        f_gate = chunks[1],
        c_gate = chunks[2],
        o_gate = chunks[3];
    // i_gate = at::sigmoid(i_gate);
    // f_gate = at::sigmoid(f_gate);
    // c_gate = at::tanh(c_gate);
    // o_gate = at::sigmoid(o_gate);
    auto C = f_gate * c_prev + i_gate * c_gate; 
    auto H = o_gate * at::tanh(C);
    return {H, C};
}