// PIMC (Perfect Information Monte Carlo) + Max^N MCTS Paodekuai AI.
//
// C++ port of the paodekuai_ai (Java) decision scheme, operating strictly on
// OUR rule engine:
//   1. Candidates always come from the real engine's legal set (must-beat,
//      first-move-3, report-single topping are enforced by the Game).
//   2. Determinization: sample full worlds consistent with the acting
//      player's view (own hand + opponents' card COUNTS + played history).
//   3. Per world, a Max^N UCB MCTS tree with greedy fast rollouts.
//   4. Aggregate root statistics across worlds; pick the candidate with the
//      best average win rate (ties broken by visits).
//
// Strength scales with `worlds` x `iterations` (deterministic given `seed`).
#pragma once

#include <vector>

#include "pdk/cards.h"
#include "pdk/game.h"
#include "pdk/moves.h"

namespace pdk {

struct PimcConfig {
    int worlds = 16;          // determinized worlds per decision
    int iterations = 400;     // MCTS iterations per world
    double exploration = 1.4; // UCB1 constant
    int rolloutDepth = 90;    // rollout ply cap (eval by fewest cards left)
    uint64_t seed = 1;        // deterministic sampling
};

// Named strength presets (planning-rollout MCTS, see README):
//   light ~7 ms/decision, standard ~50 ms, heavy ~180 ms (single core).
inline PimcConfig PimcLight()   { return PimcConfig{8, 100, 1.4, 90, 1}; }
inline PimcConfig PimcStandard(){ return PimcConfig{16, 400, 1.4, 90, 1}; }
inline PimcConfig PimcHeavy()   { return PimcConfig{48, 800, 1.4, 90, 1}; }

class PimcAi {
public:
    struct MoveStat {
        CardSet move;
        long long visits = 0;
        double avgWinRate = 0.0;
        int samples = 0;  // worlds in which the move was visited at the root
    };

    explicit PimcAi(const PimcConfig& cfg) : cfg_(cfg), rng_(cfg.seed ? cfg.seed : 0x5deece66dULL) {}

    // Restarts the determinization stream (e.g. one stream per game).
    void reseed(uint64_t seed) { rng_ = Rng(seed ? seed : 0x9e3779b97f4a7c15ULL); }

    // Returns the ranked candidate list (best first). Falls back to the
    // cheap greedy policy when the search budget is exhausted without any
    // visit, or trivially on forced/finishing moves.
    std::vector<MoveStat> decide(const Game& g, int seat);

    CardSet selectMove(const Game& g, int seat);

    const PimcConfig& config() const { return cfg_; }

private:
    PimcConfig cfg_;
    Rng rng_;
};

}  // namespace pdk
