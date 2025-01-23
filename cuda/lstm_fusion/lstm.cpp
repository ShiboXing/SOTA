#include <torch/extension.h>

#include <iostream>
#include <vector>

using namespace std;


std::vector<torch::Tensor> lstm_cell_act_forward(
    torch::Tensor i_gate,
    torch::Tensor f_gate,
    torch::Tensor c_gate,
    torch::Tensor o_gate);

namespace torch_aye {

vector<at::Tensor> lstm_cell_act_forward(
    torch::Tensor &i_gate,
    torch::Tensor &f_gate,
    torch::Tensor &c_gate,
    torch::Tensor &o_gate,
    torch::Tensor &c_prev)
{
    // cout << "i_gate: " << i_gate << "\n";
    i_gate = torch::sigmoid(i_gate);
    f_gate = torch::sigmoid(f_gate);
    c_gate = torch::tanh(c_gate);
    o_gate = torch::sigmoid(o_gate);
    auto C = f_gate * c_prev + i_gate * c_gate; 
    auto H = o_gate * torch::tanh(C);
    return {H, C};
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lstm_cell_act_forward", &torch_aye::lstm_cell_act_forward, "LSTM cell forward activation");
}