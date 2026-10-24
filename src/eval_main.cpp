// Standalone evaluation: loads a trained actor checkpoint and plays it against
// built-in opponents:
//   pimc (default) — the PIMC + Max^N MCTS AI (DouZero-style external anchor)
//   rule           — the strengthened Oracle-planning rule agent
//
// Examples:
//   perfectpao_eval --ckpt ckpt --decks 100 --opponent pimc --worlds 16 --iters 400
//   perfectpao_eval --ckpt ckpt --decks 300 --opponent rule
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "algo/eval_common.h"
#include "algo/eval_pimc.h"
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
    std::string opponent = argValue(argc, argv, "--opponent", "pimc");  // pimc | rule
    int decks = std::atoi(argValue(argc, argv, "--decks", "300"));
    int hidden = std::atoi(argValue(argc, argv, "--hidden", "256"));
    int lstmHidden = std::atoi(argValue(argc, argv, "--lstm-hidden", "128"));
    uint64_t seed = std::atoll(argValue(argc, argv, "--seed", "2026"));
    int worlds = std::atoi(argValue(argc, argv, "--worlds", "16"));
    int iters = std::atoi(argValue(argc, argv, "--iters", "400"));
    int threads = std::atoi(argValue(argc, argv, "--threads", "6"));

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

    if (opponent == "rule") {
        algo::RuleEvalConfig ecfg;
        ecfg.decks = decks;
        ecfg.seed = seed;
        auto res = algo::evaluateAgainstRule(actor, ecfg);
        if (!res.ok) {
            std::fprintf(stderr, "eval failed: %s\n", res.error.c_str());
            return 1;
        }
        std::printf("vs PlanAgent | decks %d games %d | WP %.4f ADP %+.4f (random-chance WP = 0.333)\n",
                    res.decks, res.games, res.wp, res.adp);
        return 0;
    }

    algo::PimcEvalConfig pcfg;
    pcfg.decks = decks;
    pcfg.seed = seed;
    pcfg.pimc.worlds = worlds;
    pcfg.pimc.iterations = iters;
    pcfg.threads = threads;
    auto res = algo::evaluateAgainstPimc(actor, pcfg);
    if (!res.ok) {
        std::fprintf(stderr, "eval failed: %s\n", res.error.c_str());
        return 1;
    }
    std::printf("vs PIMC(%dx%d) | decks %d games %d | WP %.4f ADP %+.4f (random-chance WP = 0.333)\n",
                worlds, iters, res.decks, res.games, res.wp, res.adp);
    return 0;
}
