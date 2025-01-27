#include <torch/extension.h>

#include <iostream>
#include <vector>

using namespace std;

namespace cuda_aye {

std::vector<torch::Tensor> lstm_cell_act_forward(
    torch::Tensor gates);

}

namespace torch_aye {

vector<at::Tensor> lstm_cell_act_forward(
    torch::Tensor &gates,
    torch::Tensor &c_prev)
{
    // cout << "i_gate: " << i_gate << "\n";
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

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lstm_cell_act_forward", &torch_aye::lstm_cell_act_forward, "LSTM cell forward activation");
}