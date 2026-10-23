#pragma once

#include <cmath>
#include <vector>

#include "algo/rollout.h"

namespace algo {

struct PPOStats {
    double pgLoss = 0;
    double vLoss = 0;
    double entropy = 0;
    double meanRet = 0;
    double meanAdv = 0;
    double approxKL = 0;
    double lastEpochKL = 0;
    double clipFraction = 0;
    int epochsCompleted = 0;
};

// Running mean and variance normalizer with Welford algorithm.
struct RunningNormalizer {
    double count = 0.0;
    double mean = 0.0;
    double M2 = 0.0;

    void update(float val) {
        count += 1.0;
        double delta = double(val) - mean;
        mean += delta / count;
        double delta2 = double(val) - mean;
        M2 += delta * delta2;
    }

    float normalize(float val) const {
        double var = (count > 1.0) ? (M2 / (count - 1.0)) : 1.0;
        float std = float(std::sqrt(std::max(var, 1e-6)));
        return float((double(val) - mean) / std);
    }

    float denormalize(float val) const {
        double var = (count > 1.0) ? (M2 / (count - 1.0)) : 1.0;
        float std = float(std::sqrt(std::max(var, 1e-6)));
        return float(double(val) * std + mean);
    }
};

// GAE / episode-return utilities (game-agnostic, same scheme as PerfectDouCPP).

// Computes GAE advantages and returns for each transition in the trajectory
// stream, chaining per gameId. When lambda == 1.0 and gamma == 1.0 this is the
// full Monte Carlo return.
void assignEpisodeReturns(std::vector<Transition>& tr, float gamma = 1.0f,
                          float lambda = 0.95f);

// Computes masked softmax over action logits, producing valid probabilities
// with NaN / Inf fallback to uniform.
void maskedSoftmax(const float* logits, int n, float* probs);

}  // namespace algo
