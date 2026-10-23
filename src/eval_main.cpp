// Standalone evaluation: loads a trained actor checkpoint and plays it against
// two heuristic RuleAgents over random decks.
//
// Example:
//   perfectpao_eval --ckpt ckpt --decks 500
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "algo/eval_rule.h"
#include "pdk/game.h"
#include "torch_nn/models.h"

namespace {

const char* argValue(int argc, char** argv, const char* key, const char* def) {
    std::string k = key;
    for (int i = 1; i + 1 < argc; ++i)
        if (k == argv[i]) return argv[i + 1];
    return def;
}

}  // namespace

int main(int argc, char** argv) {
    std::string ckpt = argValue(argc, argv, "--ckpt", "ckpt_long");
    int decks = std::atoi(argValue(argc, argv, "--decks", "300"));
    int hidden = std::atoi(argValue(argc, argv, "--hidden", "256"));
    int lstmHidden = std::atoi(argValue(argc, argv, "--lstm-hidden", "128"));
    uint64_t seed = std::atoll(argValue(argc, argv, "--seed", "0"));

    std::string actPath = ckpt + "/actor.pt";
    if (!std::filesystem::exists(actPath)) {
        std::fprintf(stderr, "checkpoint not found: %s\n", actPath.c_str());
        return 1;
    }

    torch_nn::ModelConfig cfg;
    cfg.hidden = hidden;
    cfg.lstmHidden = lstmHidden;
    torch_nn::Actor actor(cfg);
    actor->load(actPath);
    actor->eval();

    algo::RuleEvalConfig ecfg;
    ecfg.decks = decks;
    ecfg.seed = seed;
    ecfg.verbose = true;
    auto res = algo::evaluateAgainstRule(actor, ecfg);
    if (!res.ok) {
        std::fprintf(stderr, "eval failed: %s\n", res.error.c_str());
        return 1;
    }
    std::printf("vs RuleAgent | decks %d games %d | WP %.4f ADP %+.4f (random-chance WP = 0.333)\n",
                res.decks, res.games, res.wp, res.adp);
    return 0;
}
