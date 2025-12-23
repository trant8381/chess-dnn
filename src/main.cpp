#include "dnn.h"
#include <torch/torch.h>

int main() {
  DNN dnn = DNN();
  torch::Tensor input = torch::zeros({1, INPUT_PLANES, 8, 8});
  Eval output = dnn->forward(input);

  std::cout << output.policy << std::endl;
  std::cout << output.value << std::endl;
}