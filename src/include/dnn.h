#pragma once
#include "constants.h"
#include <ATen/core/TensorBody.h>
#include <string>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>

// the return struct on a forward pass of the whole model.
struct Eval {
  torch::Tensor value;
  torch::Tensor policy;
  bool initialized = false;
  Eval();
  Eval(torch::Tensor _value, torch::Tensor _policy) {
    value = _value;
    policy = _policy;
    initialized = true;
  }
};

class ConvBlockImpl : public torch::nn::Module {
private:
  torch::nn::Conv2d conv;
  torch::nn::BatchNorm2d batchNorm;
  torch::nn::ReLU relu;

public:
  ConvBlockImpl(int inputPlanes, int outputPlanes, int kernels, int padding)
      : conv(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(inputPlanes, outputPlanes, kernels)
                .padding(padding)
                .stride(1))),
        batchNorm(torch::nn::BatchNorm2d(outputPlanes)),
        relu(torch::nn::ReLU(true)) {

    register_module("conv", conv);
    register_module("batchNorm", batchNorm);
    register_module("relu", relu);
  }

  torch::Tensor forward(const torch::Tensor &x) {
    return relu(batchNorm(conv(x)));
  }
};

TORCH_MODULE(ConvBlock);

class ResBlockImpl : public torch::nn::Module {
private:
  ConvBlock conv1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm2d batchNorm;
  torch::nn::ReLU relu;

public:
  ResBlockImpl()
      : conv1(ConvBlock(TRUNK_CHANNELS, TRUNK_CHANNELS, 3, 1)),
        conv2(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(TRUNK_CHANNELS, TRUNK_CHANNELS, 3)
                .padding(1)
                .stride(1))),
        batchNorm(torch::nn::BatchNorm2d(TRUNK_CHANNELS)),
        relu(torch::nn::ReLU(true)) {

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("batchNorm", batchNorm);
    register_module("relu", relu);
  }

  torch::Tensor forward(const torch::Tensor &x) {
    return relu(batchNorm(conv2(conv1->forward(x))) + x);
  }
};

TORCH_MODULE(ResBlock);

class ValueHeadImpl : public torch::nn::Module {
private:
  ConvBlock conv;
  torch::nn::Flatten flatten;
  torch::nn::Linear fc1;
  torch::nn::ReLU relu;
  torch::nn::Linear fc2;
  torch::nn::Tanh tanh;

public:
  ValueHeadImpl()
      : conv(ConvBlock(TRUNK_CHANNELS, 1, 1, 0)), flatten(torch::nn::Flatten()),
        fc1(torch::nn::Linear(64, 256)), relu(torch::nn::ReLU(true)),
        fc2(torch::nn::Linear(256, 1)), tanh(torch::nn::Tanh()) {

    register_module("conv", conv);
    register_module("flatten", flatten);
    register_module("fc1", fc1);
    register_module("relu", relu);
    register_module("fc2", fc2);
    register_module("tanh", tanh);
  }

  torch::Tensor forward(const torch::Tensor &x) {
    return tanh(fc2(relu(fc1(flatten(conv->forward(x))))));
  }
};

TORCH_MODULE(ValueHead);

class PolicyHeadImpl : public torch::nn::Module {
private:
  ConvBlock conv;
  torch::nn::Flatten flatten;
  torch::nn::Linear fc;

public:
  PolicyHeadImpl()
      : conv(ConvBlock(TRUNK_CHANNELS, 2, 1, 0)), flatten(torch::nn::Flatten()),
        fc(torch::nn::Linear(128, 4672)) {
    register_module("conv", conv);
    register_module("flatten", flatten);
    register_module("fc", fc);
  }

  torch::Tensor forward(const torch::Tensor &x) {
    return fc(flatten(conv->forward(x)));
  }
};

TORCH_MODULE(PolicyHead);

class DNNImpl : public torch::nn::Module {
private:
  ConvBlock conv;
  torch::nn::ModuleList tower;
  ValueHead valueHead = ValueHead();
  PolicyHead policyHead = PolicyHead();

public:
  DNNImpl()
      : conv(ConvBlock(INPUT_PLANES, TRUNK_CHANNELS, 3, 1)),
        valueHead(ValueHead()), policyHead(PolicyHead()) {
    for (int i = 0; i < TOWER_SIZE; i++) {
      tower->push_back(ResBlock());
    }
    register_module("conv", conv);
    register_module("tower", tower);
    register_module("valueHead", valueHead);
    register_module("policyHead", policyHead);
  }

  Eval forward(const torch::Tensor &x) {
    torch::Tensor state = conv->forward(x);
    for (const auto &resBlock : *tower) {
      state = resBlock->as<ResBlockImpl>()->forward(state);
    }

    return Eval(valueHead->forward(state), policyHead->forward(state));
  }
};

TORCH_MODULE(DNN);