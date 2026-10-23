#include "torch_nn/models.h"

namespace torch_nn {

ActorImpl::ActorImpl(const ModelConfig& cfg) : config(cfg) {
    torch::nn::LSTMOptions lstmOpts(kLstmIn, cfg.lstmHidden);
    lstmOpts.batch_first(true);
    lstm = register_module("lstm", torch::nn::LSTM(lstmOpts));

    int64_t nodeDim = kImpInput + cfg.lstmHidden;
    int64_t jointDim = nodeDim + kActionInput;

    l1 = register_module("l1", torch::nn::Linear(jointDim, cfg.hidden));
    l2 = register_module("l2", torch::nn::Linear(cfg.hidden, cfg.hidden));
    l3 = register_module("l3", torch::nn::Linear(cfg.hidden, cfg.hidden));
    l4 = register_module("l4", torch::nn::Linear(cfg.hidden, cfg.hidden * 2));
    head = register_module("head", torch::nn::Linear(cfg.hidden * 2, 1));

    // Zero-initialize the final scoring layer (consistent with PerfectDou).
    torch::nn::init::zeros_(head->weight);
    torch::nn::init::zeros_(head->bias);
}

torch::Tensor ActorImpl::forward(const torch::Tensor& xImp,
                                 const torch::Tensor& seq,
                                 const torch::Tensor& actionFeat,
                                 const torch::Tensor& actionSample,
                                 const torch::Tensor& actionId) const {
    auto lstmOut = lstm->forward(seq);
    torch::Tensor hLast = std::get<0>(std::get<1>(lstmOut)).squeeze(0);

    torch::Tensor node = torch::cat({xImp, hLast}, /*dim=*/1);

    int64_t nodeDim = kImpInput + config.lstmHidden;
    torch::Tensor sampleIndices = actionSample.to(torch::kLong);

    // Decomposed linear projection for l1: projects node [B, nodeDim] and
    // actionFeat [L, actDim] separately, avoiding replicating node L times.
    torch::Tensor W_node = l1->weight.slice(1, 0, nodeDim);
    torch::Tensor W_act = l1->weight.slice(1, nodeDim);

    torch::Tensor nodeProj = torch::addmm(l1->bias, node, W_node.t());
    torch::Tensor actProj = torch::mm(actionFeat, W_act.t());
    torch::Tensor f1 = torch::relu(nodeProj.index_select(0, sampleIndices) + actProj);
    torch::Tensor f2 = torch::relu(l2->forward(f1));
    torch::Tensor f3 = torch::relu(l3->forward(f2));
    torch::Tensor f4 = torch::relu(l4->forward(f3));
    torch::Tensor scores = head->forward(f4).squeeze(-1); // [L]

    int64_t B = xImp.size(0);
    torch::Tensor logits = torch::full({B, kNumActions}, -1e9f,
                                      torch::TensorOptions().dtype(torch::kFloat32).device(xImp.device()));

    torch::Tensor aIndices = actionId.to(torch::kLong);
    logits.index_put_({sampleIndices, aIndices}, scores);

    return logits;
}

void ActorImpl::save(const std::string& path) const {
    torch::serialize::OutputArchive archive;
    torch::nn::Module::save(archive);
    archive.save_to(path);
}

void ActorImpl::load(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    torch::nn::Module::load(archive);
}

// CriticImpl
CriticImpl::CriticImpl(const ModelConfig& cfg) : config(cfg) {
    torch::nn::LSTMOptions lstmOpts(kLstmIn, cfg.lstmHidden);
    lstmOpts.batch_first(true);
    lstm = register_module("lstm", torch::nn::LSTM(lstmOpts));

    int64_t nodeDim = kImpInput + cfg.lstmHidden;
    int64_t allDim = nodeDim + kExtraInput;

    c1 = register_module("c1", torch::nn::Linear(allDim, cfg.hidden));
    c2 = register_module("c2", torch::nn::Linear(cfg.hidden, cfg.hidden));
    c3 = register_module("c3", torch::nn::Linear(cfg.hidden, cfg.hidden));
    c4 = register_module("c4", torch::nn::Linear(cfg.hidden, cfg.hidden));
    out = register_module("out", torch::nn::Linear(cfg.hidden, 1));

    torch::nn::init::zeros_(out->weight);
    torch::nn::init::zeros_(out->bias);
}

torch::Tensor CriticImpl::forward(const torch::Tensor& xImp,
                                  const torch::Tensor& seq,
                                  const torch::Tensor& extra) const {
    auto lstmOut = lstm->forward(seq);
    torch::Tensor hLast = std::get<0>(std::get<1>(lstmOut)).squeeze(0);

    torch::Tensor node = torch::cat({xImp, hLast}, /*dim=*/1);
    torch::Tensor all = torch::cat({node, extra}, /*dim=*/1);

    torch::Tensor q1 = torch::relu(c1->forward(all));
    torch::Tensor q2 = torch::relu(c2->forward(q1));
    torch::Tensor q3 = torch::relu(c3->forward(q2));
    torch::Tensor q4 = torch::relu(c4->forward(q3));
    torch::Tensor val = out->forward(q4); // [B, 1]

    return val;
}

void CriticImpl::save(const std::string& path) const {
    torch::serialize::OutputArchive archive;
    torch::nn::Module::save(archive);
    archive.save_to(path);
}

void CriticImpl::load(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    torch::nn::Module::load(archive);
}

}  // namespace torch_nn
