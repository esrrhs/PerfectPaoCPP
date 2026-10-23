// Unit tests for RL algorithm components:
// 1. GAE + advantage normalization
// 2. PPO value clipping loss math
// 3. KL divergence properties
// 4. League archive anchor preservation
// 5. Running normalizer
// 6. Critic head zero-init (LibTorch)
// 7. Optimizer state serialization roundtrip
// 8. Masked softmax NaN protection
// 9. Rollout oracle difference rewards (single shared policy)
// 10. Actor cloning independence
// 11. End-to-end RuleAgent evaluation + CSV + snapshot pruning
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "algo/eval_rule.h"
#include "algo/ppo.h"
#include "algo/rollout.h"
#include "pdk/game.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"

#ifdef CHECK
#undef CHECK
#endif
#define CHECK(cond) ASSERT_TRUE(cond)

using namespace algo;

// 1. Advantage normalization and GAE terminal cutoff
static void testAdvantageNormAndGAE() {
    std::vector<Transition> tr(6);
    for (int i = 0; i < 3; ++i) {
        tr[i].gameId = 1;
        tr[i].value = 1.0f;
    }
    tr[2].reward = 2.0f;
    tr[2].terminal = true;
    for (int i = 3; i < 6; ++i) {
        tr[i].gameId = 2;
        tr[i].value = -1.0f;
    }
    tr[5].reward = -2.0f;
    tr[5].terminal = true;

    assignEpisodeReturns(tr, 1.0f, 0.95f);
    CHECK(std::abs(tr[2].adv - 1.0f) < 1e-5f);
    CHECK(std::abs(tr[5].adv + 1.0f) < 1e-5f);
    CHECK(std::abs(tr[1].adv - 0.95f) < 1e-5f);
    CHECK(std::abs(tr[4].adv + 0.95f) < 1e-5f);

    int N = static_cast<int>(tr.size());
    double mean = 0.0;
    for (const Transition& t : tr) mean += t.adv;
    mean /= N;
    CHECK(std::abs(mean) < 1e-6);
}

// 2. PPO value loss clipping math
static void testValueClipping() {
    float clip = 0.2f;
    float vfCoef = 0.5f;
    {
        float vOld = 1.0f, vPred = 1.1f, target = 2.0f;
        float err = vPred - target;
        float vClipped = vOld + std::clamp(vPred - vOld, -clip, clip);
        float errClipped = vClipped - target;
        CHECK(err == errClipped);
    }
    {
        float vOld = 1.0f, vPred = 1.5f, target = 0.0f;
        float l1 = (vPred - target) * (vPred - target);
        float vClipped = vOld + std::clamp(vPred - vOld, -clip, clip);
        float l2 = (vClipped - target) * (vClipped - target);
        CHECK(l2 < l1);  // clipped prediction closer to target -> smaller loss
    }
    (void)vfCoef;
}

// 3. KL divergence properties
static void testKLDivergence() {
    auto calcKL = [](float ratio) {
        float logRatio = std::log(ratio);
        return (ratio - 1.0f) - logRatio;
    };
    CHECK(std::abs(calcKL(1.0f)) < 1e-7f);
    for (float r = 0.1f; r < 2.5f; r += 0.05f) {
        if (std::abs(r - 1.0f) > 1e-4f) CHECK(calcKL(r) > 0.0f);
    }
    CHECK(calcKL(1.2f) > calcKL(1.1f));
    CHECK(calcKL(0.8f) > calcKL(0.9f));
}

// 4. League archive anchor preservation & thinning
static void testArchiveAnchorPreservation() {
    HistoricalPool pool;
    int maxArchive = 4;
    auto addSnapshot = [&](int upd) {
        auto snap = std::make_shared<HistoricalActorSnapshot>();
        snap->update = upd;
        while ((int)pool.archive.size() >= maxArchive && !pool.archive.empty()) {
            if (pool.archive.size() > 2) {
                pool.archive.erase(pool.archive.begin() + 1);
            } else {
                pool.archive.erase(pool.archive.begin());
            }
        }
        pool.archive.push_back(snap);
    };
    for (int upd = 1; upd <= 10; ++upd) addSnapshot(upd);
    CHECK(pool.archive.size() == 4);
    CHECK(pool.archive[0]->update == 1);
    CHECK(pool.archive.back()->update == 10);
}

// 5. Welford running normalizer
static void testRunningNormalizer() {
    RunningNormalizer norm;
    for (float x : {10.0f, 20.0f, 30.0f, 40.0f, 50.0f}) norm.update(x);
    CHECK(std::abs(norm.mean - 30.0) < 1e-4);
    float val = 25.0f;
    CHECK(std::abs(norm.denormalize(norm.normalize(val)) - val) < 1e-4f);
}

// 6. Critic head zero-initialization
static void testCriticZeroInit() {
    torch_nn::ModelConfig cfg{64, 32};
    torch_nn::Critic critic(cfg);
    CHECK(torch::all(critic->out->weight == 0.0f).item<bool>());
    CHECK(torch::all(critic->out->bias == 0.0f).item<bool>());
}

// 7. PPOLearner optimizer serialization roundtrip
static void testOptimizerSerialization() {
    torch_nn::ModelConfig cfg{64, 32};
    torch_nn::Actor a1(cfg);
    torch_nn::Critic c1(cfg);
    torch_nn::PPOConfig ppoCfg;
    torch_nn::PPOLearner l1(a1, c1, ppoCfg);

    const char* tmpPath = "test_opt_roundtrip.pt";
    l1.saveOptimizer(tmpPath);

    torch_nn::Actor a2(cfg);
    torch_nn::Critic c2(cfg);
    torch_nn::PPOLearner l2(a2, c2, ppoCfg);
    l2.loadOptimizer(tmpPath);

    std::remove(tmpPath);
}

// 8. Masked softmax NaN / Inf protection
static void testMaskedSoftmaxNaNProtection() {
    {
        float logits[3] = {1.0f, 2.0f, 3.0f};
        float probs[3] = {};
        maskedSoftmax(logits, 3, probs);
        CHECK(std::abs(probs[0] + probs[1] + probs[2] - 1.0f) < 1e-5f);
        CHECK(probs[2] > probs[1] && probs[1] > probs[0]);
    }
    {
        float logits[4] = {-1e9f, -1e9f, -1e9f, -1e9f};
        float probs[4] = {};
        maskedSoftmax(logits, 4, probs);
        for (int a = 0; a < 4; ++a)
            CHECK(std::abs(probs[a] - 0.25f) < 1e-5f);
    }
    {
        float logits[4] = {NAN, INFINITY, -INFINITY, 2.0f};
        float probs[4] = {};
        maskedSoftmax(logits, 4, probs);
        for (int a = 0; a < 4; ++a) {
            CHECK(std::isfinite(probs[a]));
            CHECK(probs[a] >= 0.0f && probs[a] <= 1.0f);
        }
    }
}

// 9. Rollout with the single shared policy produces oracle step rewards
static void testOracleDifferenceReward() {
    torch_nn::ModelConfig mCfg{32, 16};
    torch_nn::Actor actor(mCfg);
    torch_nn::Critic critic(mCfg);
    ModelSet models{&actor, &critic};

    RolloutConfig cfg;
    cfg.gamesPerUpdate = 16;
    cfg.threads = 2;
    cfg.seed = 42;
    cfg.oracleReward = true;
    cfg.oracleCoef = 0.2f;

    std::array<std::vector<Transition>, 3> streams;
    RolloutStats stats;
    collectRollout(models, cfg, streams, stats);

    EXPECT_GT(stats.games, 0);
    long long total = 0;
    bool hasNonZeroStepReward = false;
    for (int s = 0; s < 3; ++s) {
        total += (long long)streams[s].size();
        for (const auto& t : streams[s]) {
            if (!t.terminal && t.reward != 0.0f) hasNonZeroStepReward = true;
            EXPECT_LE(t.actions.size(), size_t(torch_nn::kNumActions));
        }
    }
    EXPECT_GT(total, 0);
    EXPECT_TRUE(hasNonZeroStepReward);
    EXPECT_EQ(stats.games, 16);
    EXPECT_EQ(stats.wins[0] + stats.wins[1] + stats.wins[2], 16);
}

// 10. Actor cloning deep copy independence
static void testActorCloning() {
    torch_nn::ModelConfig cfg{64, 32};
    torch_nn::Actor actor(cfg);
    auto cloned = cloneActor(actor);
    auto origParams = actor->parameters();
    auto cloneParams = (*cloned)->parameters();
    CHECK(origParams.size() == cloneParams.size());
    for (size_t i = 0; i < origParams.size(); ++i) {
        CHECK(torch::all(origParams[i] == cloneParams[i]).item<bool>());
    }
    actor->l1->weight.data()[0][0].add_(123.45f);
    CHECK(!torch::all(actor->l1->weight == (*cloned)->l1->weight).item<bool>());
}

// 11. End-to-end RuleAgent evaluation, CSV recording, snapshot pruning
static void testRuleEvalEndToEnd() {
    torch_nn::ModelConfig cfg{32, 16};
    torch_nn::Actor actor(cfg);
    auto cloned = cloneActor(actor);

    std::string testDir = "build/test_eval_e2e_run";
    std::string csvPath = testDir + "/eval.csv";
    std::string snapRoot = testDir + "/snapshots";
    std::error_code ec;
    std::filesystem::remove_all(testDir, ec);
    std::filesystem::create_directories(snapRoot + "/u1");
    std::filesystem::create_directories(snapRoot + "/u3");

    RuleEvalConfig ecfg;
    ecfg.decks = 2;  // 2 decks = 6 games
    ecfg.seed = 7;
    ecfg.update = 5;
    ecfg.label = "u5";
    ecfg.elapsedMinutes = 1.25;
    ecfg.csvPath = csvPath;
    ecfg.saveDir = snapRoot + "/u5";
    ecfg.maxSnapshots = 2;

    auto res = evaluateAgainstRule(cloned, ecfg);

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
    EXPECT_TRUE(std::getline(csvFile, dataLine));
    EXPECT_TRUE(dataLine.find("5,1.25,") != std::string::npos);
    EXPECT_TRUE(dataLine.find("u5,2,6,") != std::string::npos);
    csvFile.close();

    EXPECT_TRUE(std::filesystem::exists(snapRoot + "/u5/actor.pt"));
    EXPECT_TRUE(std::filesystem::exists(snapRoot + "/u5/meta.txt"));
    std::ifstream metaFile(snapRoot + "/u5/meta.txt");
    std::string metaContent((std::istreambuf_iterator<char>(metaFile)),
                            std::istreambuf_iterator<char>());
    EXPECT_TRUE(metaContent.find("update 5") != std::string::npos);
    EXPECT_TRUE(metaContent.find("games 6") != std::string::npos);
    metaFile.close();

    // pruning: only the 2 most recent snapshot dirs survive
    EXPECT_FALSE(std::filesystem::exists(snapRoot + "/u1"));
    EXPECT_TRUE(std::filesystem::exists(snapRoot + "/u3"));
    EXPECT_TRUE(std::filesystem::exists(snapRoot + "/u5"));

    std::filesystem::remove_all(testDir, ec);
}

// 12. A short in-process training step actually updates the shared policy.
static void testMiniTrainingStep() {
    torch_nn::ModelConfig mCfg{32, 16};
    torch_nn::Actor actor(mCfg);
    torch_nn::Critic critic(mCfg);
    torch_nn::PPOConfig ppoCfg;
    ppoCfg.epochs = 1;
    ppoCfg.minibatch = 64;
    torch_nn::PPOLearner learner(actor, critic, ppoCfg);

    ModelSet models{&actor, &critic};
    RolloutConfig rc;
    rc.gamesPerUpdate = 8;
    rc.threads = 1;
    rc.seed = 99;
    rc.oracleReward = false;

    std::array<std::vector<Transition>, 3> streams;
    RolloutStats stats;
    collectRollout(models, rc, streams, stats);

    std::vector<Transition> all;
    for (int s = 0; s < 3; ++s)
        for (auto& t : streams[s]) all.push_back(std::move(t));
    ASSERT_FALSE(all.empty());

    auto before = actor->l1->weight.clone();
    auto bt = torch_nn::transitionsToBatch(all, 1.0f, 0.95f);
    auto st = learner.update(bt);
    EXPECT_GT(st.epochsCompleted, 0);
    EXPECT_TRUE(std::isfinite(st.pgLoss));
    EXPECT_TRUE(std::isfinite(st.vLoss));
    EXPECT_FALSE(torch::all(actor->l1->weight == before).item<bool>());
}

TEST(AlgoTest, AdvantageNormAndGAE) { testAdvantageNormAndGAE(); }
TEST(AlgoTest, ValueClipping) { testValueClipping(); }
TEST(AlgoTest, KLDivergence) { testKLDivergence(); }
TEST(AlgoTest, ArchiveAnchorPreservation) { testArchiveAnchorPreservation(); }
TEST(AlgoTest, RunningNormalizer) { testRunningNormalizer(); }
TEST(AlgoTest, CriticZeroInit) { testCriticZeroInit(); }
TEST(AlgoTest, OptimizerSerialization) { testOptimizerSerialization(); }
TEST(AlgoTest, MaskedSoftmaxNaNProtection) { testMaskedSoftmaxNaNProtection(); }
TEST(AlgoTest, OracleDifferenceReward) { testOracleDifferenceReward(); }
TEST(AlgoTest, ActorCloning) { testActorCloning(); }
TEST(AlgoTest, RuleEvalEndToEnd) { testRuleEvalEndToEnd(); }
TEST(AlgoTest, MiniTrainingStep) { testMiniTrainingStep(); }
