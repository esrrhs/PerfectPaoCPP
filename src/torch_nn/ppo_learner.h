#pragma once

#include "torch_nn/models.h"
#include <torch/torch.h>

namespace torch_nn {

struct PPOConfig {
    float lr = 3e-4f;
    float clip = 0.2f;
    float entCoef = 0.03f;
    float vfCoef = 0.5f;
    float maxGradNorm = 0.5f;
    float criticMaxGradNorm = 5.0f;
    int epochs = 2;
    int minibatch = 2048;
    float targetKL = 0.03f;
};

struct BatchTensors {
    torch::Tensor xImp;          // [B, 1676]
    torch::Tensor seq;           // [B, 10, 540]
    torch::Tensor actionFeat;    // [L, 186]
    torch::Tensor actionSample;  // [L]
    torch::Tensor actionId;      // [L]
    torch::Tensor extra;         // [B, 362]
    torch::Tensor actions;       // [B] (int64)
    torch::Tensor oldLogp;       // [B] (float32)
    torch::Tensor adv;           // [B] (float32)
    torch::Tensor ret;           // [B] (float32)
    torch::Tensor oldValue;      // [B] (float32)
};

struct PPOStats {
    double pgLoss = 0;
    double vLoss = 0;
    double entropy = 0;
    double kl = 0;
    double clipFraction = 0;
    int epochsCompleted = 0;
};

class PPOLearner {
public:
    PPOLearner(Actor actor, Critic critic, const PPOConfig& cfg, torch::Device device = torch::kCPU);

    void setLr(float lr);
    void setEntCoef(float entCoef);

    PPOStats update(BatchTensors& batch);

    void saveOptimizer(const std::string& path) const;
    void loadOptimizer(const std::string& path);
    void saveOptimizer(const std::string& actorOptPath, const std::string& criticOptPath) const;
    void loadOptimizer(const std::string& actorOptPath, const std::string& criticOptPath);

    Actor actor;
    Critic critic;
    PPOConfig config;
    torch::Device device;
    std::unique_ptr<torch::optim::Adam> actorOpt;
    std::unique_ptr<torch::optim::Adam> criticOpt;
};

}  // namespace torch_nn
