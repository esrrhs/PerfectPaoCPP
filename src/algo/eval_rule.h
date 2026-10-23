// Built-in evaluation against the heuristic RuleAgent — the Paodekuai
// counterpart of PerfectDouCPP's DouZero evaluation loop, minus the external
// TCP server (no public Paodekuai baseline exists).
//
// Protocol: for every deck, three games are played with the policy occupying
// each seat in turn (the two other seats are RuleAgents). The policy picks
// greedy argmax actions (deterministic evaluation).
#pragma once

#include <string>

#include "pdk/cards.h"
#include "torch_nn/models.h"

namespace algo {

struct RuleEvalConfig {
    int decks = 200;              // decks per evaluation (3 games per deck)
    uint64_t seed = 0;            // 0 means auto seed
    std::string csvPath = "";     // if not empty, append to CSV
    std::string saveDir = "";     // if not empty, save model + meta to disk
    int maxSnapshots = 0;         // 0 = keep all, >0 = keep most recent N
    std::string label = "";
    int update = 0;
    double elapsedMinutes = 0.0;
    bool verbose = false;
};

struct RuleEvalResult {
    bool ok = false;
    int decks = 0;
    int games = 0;
    double wp = 0.0;   // policy win rate
    double adp = 0.0;  // mean normalized payoff per game
    std::string error;
};

// Deep copies the actor into a heap-allocated snapshot.
std::shared_ptr<torch_nn::Actor> cloneActor(torch_nn::Actor src);

// Evaluates the actor against two RuleAgents over `cfg.decks` random decks.
RuleEvalResult evaluateAgainstRule(torch_nn::Actor& actor,
                                   const RuleEvalConfig& cfg);
RuleEvalResult evaluateAgainstRule(std::shared_ptr<torch_nn::Actor> actor,
                                   const RuleEvalConfig& cfg);

}  // namespace algo
