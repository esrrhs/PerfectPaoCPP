// Evaluation against the built-in PIMC + Max^N MCTS AI — the Paodekuai
// counterpart of PerfectDouCPP's DouZero evaluation loop.
//
// Protocol mirrors the RuleAgent evaluation: for every deck, three games are
// played with the policy occupying each seat in turn; the two other seats are
// PIMC AIs at the configured strength tier. The policy picks greedy argmax
// actions. Games run in parallel across a small thread pool; model forwards
// are serialized by a mutex (they are a negligible fraction of the cost).
#pragma once

#include <memory>
#include <string>

#include "pdk/cards.h"
#include "pdk/pimc_ai.h"
#include "torch_nn/models.h"

namespace algo {

struct PimcEvalConfig {
    int decks = 100;             // decks per evaluation (3 games per deck)
    uint64_t seed = 0;           // 0 means auto seed
    pdk::PimcConfig pimc{16, 400, 1.4, 90, 1};  // strength tier
    int threads = 4;             // parallel evaluation games
    std::string csvPath = "";    // if not empty, append to CSV
    std::string saveDir = "";    // if not empty, save model + meta to disk
    int maxSnapshots = 0;        // 0 = keep all, >0 = keep most recent N
    std::string label = "";
    int update = 0;
    double elapsedMinutes = 0.0;
};

struct PimcEvalResult {
    bool ok = false;
    int decks = 0;
    int games = 0;
    double wp = 0.0;
    double adp = 0.0;
    std::string error;
};

// Evaluates the actor against two PIMC AIs per game over `cfg.decks` decks.
PimcEvalResult evaluateAgainstPimc(torch_nn::Actor& actor,
                                   const PimcEvalConfig& cfg);
PimcEvalResult evaluateAgainstPimc(std::shared_ptr<torch_nn::Actor> actor,
                                   const PimcEvalConfig& cfg);

}  // namespace algo
