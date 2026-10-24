// Unit tests for the strengthened planning agent, the PIMC + Max^N MCTS AI,
// the rule-opponent rollout path, and the PIMC evaluation harness.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include "algo/eval_pimc.h"
#include "algo/eval_rule.h"
#include "algo/rollout.h"
#include "pdk/game.h"
#include "pdk/pimc_ai.h"
#include "pdk/rule_agent.h"
#include "torch_nn/models.h"

#ifdef CHECK
#undef CHECK
#endif
#define CHECK(cond) ASSERT_TRUE(cond)

using namespace pdk;

namespace {

CardSet randomAgent(const std::vector<CardSet>& legal, Rng& rng) {
    return legal[rng.below(int(legal.size()))];
}

}  // namespace

// The strengthened planning agent comfortably beats uniformly random play.
TEST(PimcTest, PlanAgentBeatsRandom) {
    Rng rng(4242);
    const int games = 300;
    int wins = 0;
    for (int g = 0; g < games; ++g) {
        Game game;
        game.deal(rng);
        int seat = g % 3;
        while (!game.over) {
            std::vector<CardSet> legal = game.legal();
            CardSet m = (game.turn == seat) ? PlanAgent::selectMove(game, seat, legal)
                                            : randomAgent(legal, rng);
            game.step(m);
        }
        if (game.winner == seat) ++wins;
    }
    double wp = double(wins) / games;
    std::printf("  PlanAgent vs random WP %.3f\n", wp);
    CHECK(wp > 0.60);
}

// And it clearly dominates the cheap greedy policy in a mixed match
// (one PlanAgent seat vs two greedy seats).
TEST(PimcTest, PlanAgentBeatsGreedy) {
    Rng rng(1717);
    const int games = 300;
    int wins = 0;
    for (int g = 0; g < games; ++g) {
        Game game;
        game.deal(rng);
        int seat = g % 3;
        while (!game.over) {
            std::vector<CardSet> legal = game.legal();
            CardSet m = (game.turn == seat) ? PlanAgent::selectMove(game, seat, legal)
                                            : RuleAgent::selectMove(game, game.turn, legal);
            game.step(m);
        }
        if (game.winner == seat) ++wins;
    }
    double wp = double(wins) / games;
    std::printf("  PlanAgent vs 2x greedy WP %.3f\n", wp);
    CHECK(wp > 0.40);
}

// PIMC always picks engine-legal moves and is deterministic under a fixed
// seed, even across re-instantiated searches.
TEST(PimcTest, LegalityAndDeterminism) {
    PimcConfig cfg{3, 20, 1.4, 90, 777};

    auto playTrace = [&](uint64_t aiSeed) {
        Rng rng(99);
        std::vector<std::string> trace;
        Game game;
        game.deal(rng);
        int aiSeat = 0;
        PimcAi ai(cfg);
        ai.reseed(aiSeed);
        while (!game.over) {
            std::vector<CardSet> legal = game.legal();
            EXPECT_TRUE(!legal.empty());
            if (game.turn == aiSeat) {
                CardSet m = ai.selectMove(game, aiSeat);
                bool ok = false;
                for (const CardSet& l : legal)
                    if (l == m) ok = true;
                EXPECT_TRUE(ok);  // legality under the full engine constraint set
                trace.push_back(m.str());
                game.step(m);
            } else {
                CardSet m = RuleAgent::selectMove(game, game.turn, legal);
                trace.push_back(m.str());
                game.step(m);
            }
        }
        return trace;
    };

    auto t1 = playTrace(555);
    auto t2 = playTrace(555);
    CHECK(t1 == t2);
    CHECK(t1.size() > 5);

    auto t3 = playTrace(556);
    // a different stream may legitimately differ somewhere
    (void)t3;

    // ranked list contract: non-empty, first entry visited
    Rng rng(5);
    Game g;
    g.deal(rng);
    PimcAi ai(cfg);
    ai.reseed(9);
    auto ranked = ai.decide(g, g.turn);
    CHECK(!ranked.empty());
    CHECK(ranked.front().visits > 0);
}

// Tiny-budget PIMC still beats random play decisively.
TEST(PimcTest, PimcBeatsRandom) {
    Rng rng(31337);
    PimcConfig cfg{4, 60, 1.4, 90, 1};
    const int games = 15;
    int wins = 0;
    for (int g = 0; g < games; ++g) {
        Game game;
        game.deal(rng);
        int seat = g % 3;
        PimcAi ai(cfg);
        ai.reseed(uint64_t(g) * 7919 + 3);
        while (!game.over) {
            std::vector<CardSet> legal = game.legal();
            CardSet m;
            if (game.turn == seat) {
                m = ai.selectMove(game, seat);
            } else {
                m = randomAgent(legal, rng);
            }
            game.step(m);
        }
        if (game.winner == seat) ++wins;
    }
    double wp = double(wins) / games;
    std::printf("  PIMC(4x60) vs random WP %.3f\n", wp);
    CHECK(wp >= 0.4);
}

// Rollout with every game facing the rule agent still completes and produces
// learning transitions for the two latest seats.
TEST(PimcTest, RolloutWithRuleOpponents) {
    torch_nn::ModelConfig mCfg{32, 16};
    torch_nn::Actor actor(mCfg);
    torch_nn::Critic critic(mCfg);
    algo::ModelSet models{&actor, &critic};

    algo::RolloutConfig cfg;
    cfg.gamesPerUpdate = 12;
    cfg.threads = 2;
    cfg.seed = 7;
    cfg.historicalProb = 0.0f;
    cfg.ruleOpponentProb = 1.0f;  // every game: one rule seat, two latest

    std::array<std::vector<algo::Transition>, 3> streams;
    algo::RolloutStats stats;
    algo::collectRollout(models, cfg, streams, stats);

    EXPECT_EQ(stats.games, 12);
    long long total = 0;
    for (int s = 0; s < 3; ++s) total += (long long)streams[s].size();
    // two learning seats per game, each with several decisions
    EXPECT_GT(total, 12 * 10);
}

// End-to-end PIMC evaluation harness: CSV + snapshot + pruning.
TEST(PimcTest, EvalPimcEndToEnd) {
    torch_nn::ModelConfig cfg{32, 16};
    torch_nn::Actor actor(cfg);
    auto cloned = algo::cloneActor(actor);

    std::string testDir = "build/test_pimc_e2e_run";
    std::string csvPath = testDir + "/eval.csv";
    std::string snapRoot = testDir + "/snapshots";
    std::error_code ec;
    std::filesystem::remove_all(testDir, ec);
    std::filesystem::create_directories(snapRoot + "/u1");

    algo::PimcEvalConfig pcfg;
    pcfg.decks = 2;  // 6 games
    pcfg.seed = 21;
    pcfg.threads = 2;
    pcfg.pimc.worlds = 2;
    pcfg.pimc.iterations = 5;
    pcfg.update = 9;
    pcfg.label = "u9";
    pcfg.elapsedMinutes = 0.5;
    pcfg.csvPath = csvPath;
    pcfg.saveDir = snapRoot + "/u9";
    pcfg.maxSnapshots = 1;

    auto res = algo::evaluateAgainstPimc(cloned, pcfg);

    EXPECT_TRUE(res.ok);
    EXPECT_EQ(res.decks, 2);
    EXPECT_EQ(res.games, 6);
    EXPECT_GE(res.wp, 0.0);
    EXPECT_LE(res.wp, 1.0);

    EXPECT_TRUE(std::filesystem::exists(csvPath));
    std::ifstream csvFile(csvPath);
    std::string header, dataLine;
    EXPECT_TRUE(std::getline(csvFile, header));
    EXPECT_TRUE(header.find("update,minutes,timestamp") != std::string::npos);
    EXPECT_TRUE(header.find("worlds,iterations") != std::string::npos);
    EXPECT_TRUE(std::getline(csvFile, dataLine));
    EXPECT_TRUE(dataLine.find("9,0.50,") != std::string::npos);
    EXPECT_TRUE(dataLine.find(",2,6,") != std::string::npos);
    csvFile.close();

    EXPECT_TRUE(std::filesystem::exists(snapRoot + "/u9/actor.pt"));
    EXPECT_TRUE(std::filesystem::exists(snapRoot + "/u9/meta.txt"));
    std::ifstream metaFile(snapRoot + "/u9/meta.txt");
    std::string metaContent((std::istreambuf_iterator<char>(metaFile)),
                            std::istreambuf_iterator<char>());
    EXPECT_TRUE(metaContent.find("update 9") != std::string::npos);
    EXPECT_TRUE(metaContent.find("worlds 2") != std::string::npos);
    metaFile.close();
    EXPECT_FALSE(std::filesystem::exists(snapRoot + "/u1"));  // pruned

    std::filesystem::remove_all(testDir, ec);
}
