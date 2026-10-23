#pragma once

#include "algo/rollout.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include <torch/torch.h>

namespace torch_nn {

// Convert standard C++ Transitions into continuous PyTorch tensors
// (GAE + advantage normalization included).
BatchTensors transitionsToBatch(const std::vector<algo::Transition>& tr,
                                float gamma = 1.0f,
                                float lambda = 0.95f,
                                torch::Device device = torch::kCPU);

}  // namespace torch_nn
