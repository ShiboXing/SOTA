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
    const int64_t numel, 
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
  if (i >= numel) return;
  const int64_t col = i % state_size;
  const int64_t tanh_st = 2 * state_size, tanh_ed = tanh_st + state_size;
  if (col >= tanh_st && col < tanh_ed) {
    gates[i] = tanh(gates[i]);
  } else {
    gates[i] = sigmoid(gates[i]);
  }
  // if (column < state_size) {
  //   input_gate[index] = sigmoid(gates[gates_row + column]);
  //   output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
  //   candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
  //   new_cell[index] =
  //       old_cell[index] + candidate_cell[index] * input_gate[index];
  //   new_h[index] = tanh(new_cell[index]) * output_gate[index];
  // }
}

vector<at::Tensor> lstm_cell_act_forward_cuda(
    torch::Tensor &gates,
    torch::Tensor &c_prev)
{
    // cout << "i_gate: " << i_gate << "\n";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threads = prop.maxThreadsPerBlock;
    const dim3 blocks((gates.numel() + threads - 1) / threads);
    // cout << "threads per block called:" << threads << "\n";
    // std::cout << "Blocks called: (" << blocks.x << ", " << blocks.y << ", " << blocks.z << ")" << std::endl;

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lstm_cell_act_forward", ([&] {
        lstm_cell_act_fwd<scalar_t><<<blocks, threads>>>(
            gates.data<scalar_t>(), gates.numel(), c_prev.size(-1));
    }));
    
    vector<torch::Tensor> chunks = torch::chunk(gates, 4, 2);
    torch::Tensor i_gate = chunks[0], 
        f_gate = chunks[1],
        c_gate = chunks[2],
        o_gate = chunks[3];
    i_gate = at::sigmoid(i_gate);
    f_gate = at::sigmoid(f_gate);
    c_gate = at::tanh(c_gate);
    o_gate = at::sigmoid(o_gate);
    auto C = f_gate * c_prev + i_gate * c_gate; 
    auto H = o_gate * at::tanh(C);
    return {H, C};
}