#include <torch/extension.h>

#include <iostream>
#include <vector>

using namespace std;

vector<at::Tensor> lstm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell)
{
    return {};
}