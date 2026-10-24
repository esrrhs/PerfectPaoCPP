// Batched self-play rollout collection for Paodekuai.
//
// A pool of games is advanced in lockstep. Because the three seats are fully
// symmetric and share ONE policy, every game waiting on the current player is
// evaluated in a single batched forward pass regardless of seat. Transitions
// are still split into per-seat streams so GAE credit assignment chains within
// one seat's trajectory; the learner later merges the streams.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "pdk/features.h"
#include "pdk/game.h"
#include "torch_nn/models.h"

namespace algo {

struct Transition {
    std::array<uint8_t, pdk::kImpBin> imp{};
    std::array<float, pdk::kImpScalars> scalar{};
    std::array<uint8_t, pdk::kExtraBin> extra{};
    std::array<float, pdk::kExtraScalars> extraScalar{};
    // Action representation (local slot -> 11x13 + 6).
    std::vector<std::pair<int, std::array<float, pdk::kActionSize>>> actions;
    int action = -1;
    int gameId = -1;
    bool terminal = false;
    float logp = 0.0f;
    float value = 0.0f;
    float reward = 0.0f;
    float adv = 0.0f;
    float ret = 0.0f;
};

struct RolloutStats {
    int games = 0;
    std::array<long long, 3> wins{};   // first-out wins per seat (symmetry check)
    double winnerMargin = 0.0;         // sum of normalized winner payoffs
    int pureSelfPlayGames = 0;
    double pureSelfPlayMargin = 0.0;
    long long bombs = 0;
    long long moves = 0;
    std::array<long long, 3> transitions{};
};

// One shared policy for all three symmetric seats.
struct ModelSet {
    torch_nn::Actor* actor = nullptr;
    torch_nn::Critic* critic = nullptr;
};

struct HistoricalActorSnapshot {
    torch_nn::Actor actor;
    int update = 0;  // update index at which this snapshot was created
};

// Hierarchical league pool: recent rolling snapshots + geometrically spaced
// long-term archive (same scheme as PerfectDouCPP).
struct HistoricalPool {
    std::vector<std::shared_ptr<HistoricalActorSnapshot>> recent;
    std::vector<std::shared_ptr<HistoricalActorSnapshot>> archive;

    bool empty() const {
        return recent.empty() && archive.empty();
    }
};

struct RolloutConfig {
    int gamesPerUpdate = 256;
    int threads = 4;
    uint64_t seed = 1;
    float historicalProb = 0.2f;  // probability of a game facing a snapshot
    const HistoricalPool* historicalPool = nullptr;
    float ruleOpponentProb = 0.0f;  // probability of a game facing the
                                    // strengthened rule agent on one seat
    // Oracle step-wise dense difference reward: per-seat advantage is
    // n(seat) - min(n(others)); acting seat earns (lastAdv - curAdv) * coef.
    bool oracleReward = true;
    float oracleCoef = 0.2f;
};

// Runs self-play / league games using the current models and historical pool.
void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 3>& out,
                    RolloutStats& stats);

}  // namespace algo
