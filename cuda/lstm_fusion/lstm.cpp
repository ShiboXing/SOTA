#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#include <iostream>
#include <vector>

using namespace std;

vector<torch::Tensor> lstm_cell_act_forward_cuda(
    torch::Tensor &input, 
    torch::Tensor &h_prev,
    torch::Tensor &c_prev,
    torch::Tensor &wi, 
    torch::Tensor &wh,
    torch::Tensor &bi,
    torch::Tensor &bh);

vector<at::Tensor> lstm_cell_act_forward(
    torch::Tensor &input, 
    torch::Tensor &h_prev,
    torch::Tensor &c_prev,
    torch::Tensor &wi, 
    torch::Tensor &wh,
    torch::Tensor &bi,
    torch::Tensor &bh) {
    CHECK_INPUT(input);
    CHECK_INPUT(h_prev);
    CHECK_INPUT(c_prev);
    CHECK_INPUT(wi);
    CHECK_INPUT(wh);
    CHECK_INPUT(bi);
    CHECK_INPUT(bh);
    return lstm_cell_act_forward_cuda(input, h_prev, c_prev, wi, wh, bi, bh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lstm_cell_act_forward", &lstm_cell_act_forward, "LSTM cell forward activation");
}