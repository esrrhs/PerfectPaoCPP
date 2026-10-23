#include "torch_nn/ppo_learner.h"
#include <cmath>
#include <filesystem>

namespace torch_nn {

PPOLearner::PPOLearner(Actor actor_, Critic critic_, const PPOConfig& cfg, torch::Device device_)
    : actor(actor_), critic(critic_), config(cfg), device(device_) {
    actor->to(device);
    critic->to(device);

    torch::optim::AdamOptions aOpts(config.lr);
    aOpts.betas(std::make_tuple(0.9, 0.999)).eps(1e-5);
    actorOpt = std::make_unique<torch::optim::Adam>(actor->parameters(), aOpts);

    torch::optim::AdamOptions cOpts(config.lr);
    cOpts.betas(std::make_tuple(0.9, 0.999)).eps(1e-5);
    criticOpt = std::make_unique<torch::optim::Adam>(critic->parameters(), cOpts);
}

void PPOLearner::setLr(float lr) {
    config.lr = lr;
    for (auto& param_group : actorOpt->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(lr);
    }
    for (auto& param_group : criticOpt->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(lr);
    }
}

void PPOLearner::setEntCoef(float entCoef) {
    config.entCoef = entCoef;
}

PPOStats PPOLearner::update(BatchTensors& batch) {
    int64_t N = batch.xImp.size(0);
    PPOStats stats;
    if (N == 0) return stats;

    actor->train();
    critic->train();

    int64_t mbSize = std::min<int64_t>(config.minibatch, N);
    int64_t nBatches = (N + mbSize - 1) / mbSize;

    double pgLossSum = 0, vLossSum = 0, entSum = 0;
    double klSum = 0, clipCountSum = 0;
    int totalSamples = 0;

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        // Shuffle dataset indices
        torch::Tensor indices = torch::randperm(N, torch::TensorOptions().dtype(torch::kLong));
        double epochKL = 0;
        int epochSamples = 0;

        for (int64_t b = 0; b < nBatches; ++b) {
            int64_t start = b * mbSize;
            int64_t end = std::min(start + mbSize, N);
            int64_t curB = end - start;
            torch::Tensor mbIndices = indices.slice(0, start, end).to(device);

            // Subselect sample-level tensors
            torch::Tensor xImpMb = batch.xImp.index_select(0, mbIndices);
            torch::Tensor seqMb = batch.seq.index_select(0, mbIndices);
            torch::Tensor extraMb = batch.extra.index_select(0, mbIndices);
            torch::Tensor actMb = batch.actions.index_select(0, mbIndices).to(device);
            torch::Tensor oldLogpMb = batch.oldLogp.index_select(0, mbIndices).to(device);
            torch::Tensor advMb = batch.adv.index_select(0, mbIndices).to(device);
            torch::Tensor retMb = batch.ret.index_select(0, mbIndices).to(device);
            torch::Tensor oldValueMb = batch.oldValue.index_select(0, mbIndices).to(device);

            // Create sample mapping from global indices to minibatch [0..curB-1]
            // We find which action elements belong to the selected samples:
            // Create lookup table: lookup[global_sample_id] = local_sample_id
            torch::Tensor lookup = torch::full({N}, -1, torch::TensorOptions().dtype(torch::kLong).device(batch.actionSample.device()));
            lookup.index_put_({mbIndices.to(batch.actionSample.device())}, torch::arange(curB, torch::TensorOptions().dtype(torch::kLong).device(batch.actionSample.device())));

            torch::Tensor mappedSample = lookup.index_select(0, batch.actionSample.to(torch::kLong));
            torch::Tensor mask = mappedSample >= 0;
            torch::Tensor actIndices = mask.nonzero().squeeze(-1);

            if (actIndices.numel() == 0) continue;

            torch::Tensor actFeatMb = batch.actionFeat.index_select(0, actIndices).to(device);
            torch::Tensor actSampleMb = mappedSample.index_select(0, actIndices).to(device);
            torch::Tensor actIdMb = batch.actionId.index_select(0, actIndices).to(device);

            // 1. Actor Forward
            torch::Tensor logits = actor->forward(xImpMb, seqMb, actFeatMb, actSampleMb, actIdMb);
            torch::Tensor logProbs = torch::log_softmax(logits, /*dim=*/1);
            torch::Tensor probs = torch::softmax(logits, /*dim=*/1);

            // Extract chosen action logp
            torch::Tensor curLogp = logProbs.gather(1, actMb.unsqueeze(1)).squeeze(1);

            // Ratio & KL
            torch::Tensor logRatio = curLogp - oldLogpMb;
            torch::Tensor ratio = torch::exp(logRatio);

            // KL approximation: (ratio - 1) - logRatio
            torch::Tensor approxKL = (ratio - 1.0f) - logRatio;
            approxKL = torch::nan_to_num(approxKL, /*nan=*/0.0);
            double mbKL = approxKL.mean().item<double>();
            epochKL += mbKL * curB;
            epochSamples += curB;

            // Policy gradient surrogate loss
            torch::Tensor surr1 = ratio * advMb;
            torch::Tensor surr2 = torch::clamp(ratio, 1.0f - config.clip, 1.0f + config.clip) * advMb;
            torch::Tensor pgLoss = -torch::min(surr1, surr2).mean();

            // Entropy loss
            // Only sum over valid logits (> -1e8)
            torch::Tensor entropy = -(probs * torch::nan_to_num(logProbs, /*nan=*/0.0)).sum(/*dim=*/1).mean();
            torch::Tensor totalActorLoss = pgLoss - config.entCoef * entropy;

            // Actor update
            actorOpt->zero_grad();
            totalActorLoss.backward();
            torch::nn::utils::clip_grad_norm_(actor->parameters(), config.maxGradNorm);
            actorOpt->step();

            // 2. Critic Forward & Loss
            torch::Tensor values = critic->forward(xImpMb, seqMb, extraMb).squeeze(-1);
            torch::Tensor vLoss = config.vfCoef * 0.5f * (values - retMb).pow(2).mean();

            // Critic update
            criticOpt->zero_grad();
            vLoss.backward();
            torch::nn::utils::clip_grad_norm_(critic->parameters(), config.criticMaxGradNorm);
            criticOpt->step();

            // Track stats
            pgLossSum += pgLoss.item<double>() * curB;
            vLossSum += vLoss.item<double>() * curB;
            entSum += entropy.item<double>() * curB;
            klSum += mbKL * curB;
            torch::Tensor clipped = ((ratio > 1.0f + config.clip) | (ratio < 1.0f - config.clip)).to(torch::kFloat32);
            clipCountSum += clipped.sum().item<double>();
            totalSamples += curB;
        }

        stats.epochsCompleted = epoch + 1;
        // Early stop if KL divergence exceeds threshold
        if (epochSamples > 0 && (epochKL / epochSamples) > config.targetKL * 1.5) {
            break;
        }
    }

    if (totalSamples > 0) {
        stats.pgLoss = pgLossSum / totalSamples;
        stats.vLoss = vLossSum / totalSamples;
        stats.entropy = entSum / totalSamples;
        stats.kl = klSum / totalSamples;
        stats.clipFraction = clipCountSum / totalSamples;
    }

    return stats;
}

void PPOLearner::saveOptimizer(const std::string& path) const {
    torch::serialize::OutputArchive archive;
    torch::serialize::OutputArchive aArchive;
    actorOpt->save(aArchive);
    archive.write("actor_opt", aArchive);

    torch::serialize::OutputArchive cArchive;
    criticOpt->save(cArchive);
    archive.write("critic_opt", cArchive);

    archive.save_to(path);
}

void PPOLearner::loadOptimizer(const std::string& path) {
    if (!std::filesystem::exists(path)) return;
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    torch::serialize::InputArchive aArchive;
    if (archive.try_read("actor_opt", aArchive)) {
        actorOpt->load(aArchive);
    }
    torch::serialize::InputArchive cArchive;
    if (archive.try_read("critic_opt", cArchive)) {
        criticOpt->load(cArchive);
    }
}

void PPOLearner::saveOptimizer(const std::string& actorOptPath, const std::string& criticOptPath) const {
    torch::serialize::OutputArchive aArchive;
    actorOpt->save(aArchive);
    aArchive.save_to(actorOptPath);

    torch::serialize::OutputArchive cArchive;
    criticOpt->save(cArchive);
    cArchive.save_to(criticOptPath);
}

void PPOLearner::loadOptimizer(const std::string& actorOptPath, const std::string& criticOptPath) {
    if (std::filesystem::exists(actorOptPath)) {
        torch::serialize::InputArchive aArchive;
        aArchive.load_from(actorOptPath);
        actorOpt->load(aArchive);
    }
    if (std::filesystem::exists(criticOptPath)) {
        torch::serialize::InputArchive cArchive;
        cArchive.load_from(criticOptPath);
        criticOpt->load(cArchive);
    }
}

}  // namespace torch_nn
