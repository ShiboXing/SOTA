#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#include <iostream>
#include <vector>

using namespace std;

vector<torch::Tensor> lstm_cell_act_forward_cuda(
    torch::Tensor &gates,
    torch::Tensor &c_prev);

vector<at::Tensor> lstm_cell_act_forward(
    torch::Tensor &gates,
    torch::Tensor &c_prev) {
    CHECK_INPUT(gates);
    CHECK_INPUT(c_prev);
    return lstm_cell_act_forward_cuda(gates, c_prev);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lstm_cell_act_forward", &lstm_cell_act_forward, "LSTM cell forward activation");
}