#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

#include "pdk/features.h"

namespace torch_nn {

// Input dimensions derived from the Paodekuai feature scheme.
constexpr int64_t kImpInput = pdk::kNodeSize;   // 1188
constexpr int64_t kExtraInput = pdk::kExtraSize; // 288
constexpr int64_t kLstmSteps = pdk::kHistoryGroups; // 5
constexpr int64_t kLstmIn = 3 * pdk::kCardMat;  // 429 (3 moves per step)
constexpr int64_t kNumActions = 2048;
constexpr int64_t kActionInput = pdk::kActionSize; // 149

struct ModelConfig {
    int64_t hidden = 256;
    int64_t lstmHidden = 128;
};

// Actor (Imperfect Information Policy Network) — one shared model for all
// three symmetric seats.
// LSTM(429 -> lstmHidden) over 5 steps
// node = concat(xImp, lstm_h_last) -> [B, kImpInput + lstmHidden]
// joint = concat(node[actionSample], actionFeat) -> [L, nodeDim + kActionInput]
// MLP: Linear(joint -> hidden) -> ReLU -> 3x Linear(hidden -> hidden) -> ReLU
//      -> Linear(hidden -> 1)
// scatter scores [L, 1] to logits [B, kNumActions] initialized to -1e9
class ActorImpl : public torch::nn::Module {
public:
    ActorImpl() : ActorImpl(ModelConfig{}) {}
    ActorImpl(const ModelConfig& cfg);

    // seq: [B, kLstmSteps, kLstmIn]
    // xImp: [B, kImpInput]
    // actionFeat: [L, kActionInput]
    // actionSample: [L] (indices in 0..B-1)
    // actionId: [L] (slots in 0..kNumActions-1)
    // Returns logits: [B, kNumActions]
    torch::Tensor forward(const torch::Tensor& xImp,
                          const torch::Tensor& seq,
                          const torch::Tensor& actionFeat,
                          const torch::Tensor& actionSample,
                          const torch::Tensor& actionId) const;

    void save(const std::string& path) const;
    void load(const std::string& path);

    ModelConfig config;
    mutable torch::nn::LSTM lstm{nullptr};
    mutable torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, l4{nullptr}, head{nullptr};
};
TORCH_MODULE(Actor);

// Critic (Perfect Information Value Network)
// node = concat(xImp, lstm_h_last); all = concat(node, extra) -> MLP -> 1
class CriticImpl : public torch::nn::Module {
public:
    CriticImpl() : CriticImpl(ModelConfig{}) {}
    CriticImpl(const ModelConfig& cfg);

    torch::Tensor forward(const torch::Tensor& xImp,
                          const torch::Tensor& seq,
                          const torch::Tensor& extra) const;

    void save(const std::string& path) const;
    void load(const std::string& path);

    ModelConfig config;
    mutable torch::nn::LSTM lstm{nullptr};
    mutable torch::nn::Linear c1{nullptr}, c2{nullptr}, c3{nullptr}, c4{nullptr}, out{nullptr};
};
TORCH_MODULE(Critic);

}  // namespace torch_nn
