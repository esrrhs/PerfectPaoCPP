// PerfectPao CPP — Paodekuai self-play PPO training entry point.
//
// Synchronous on-policy loop (same architecture as PerfectDouCPP):
//   1. Multi-threaded CPU self-play rollout with the shared symmetric policy
//      (plus a hierarchical league of historical snapshots)
//   2. GPU/CPU PPO update over the merged per-seat transition streams
//   3. Weights synced back to CPU rollout models
//   4. Periodic full checkpoints + built-in RuleAgent evaluation (CSV/snapshots)
//
// Example:
//   perfectpao_train --updates 200 --games 256 --threads 8 --out ckpt
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <powrprof.h>
#endif

#include "algo/eval_pimc.h"
#include "algo/eval_rule.h"
#include "algo/ppo.h"
#include "algo/rollout.h"
#include "pdk/oracle.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"
#include <torch/version.h>

namespace {

struct Args {
    int updates = 100000;
    int startUpdate = 1;
    int games = 768;
    int threads = std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 16);
    int hidden = 256;
    int lstmHidden = 128;
    int epochs = 2;
    int minibatch = 2048;
    int snapshotEvery = 50;
    float lr = 3e-4f;
    float clip = 0.2f;
    float ent = 0.03f;
    float gamma = 1.0f;
    float lambda = 0.95f;
    float targetKL = 0.03f;
    bool oracleReward = true;      // per-seat oracle step difference reward
    float oracleCoef = 0.2f;
    int poolSize = 16;
    int archiveSize = 16;
    int poolEvery = 20;
    float histProb = 0.2f;
    float ruleOppProb = 0.15f;  // league games vs the strengthened rule agent
    uint64_t seed = 1;
    std::string out = "ckpt_long";
    std::string resume;
    std::string backend = "cpu";   // cuda | mps | cpu
    int evalEvery = 200;
    int evalDecks = 200;           // decks per eval; 3 games per deck
    std::string evalCsv;           // empty = <out>/eval_vs_rule.csv
    std::string evalSaveDir;       // empty = <out>/eval_snapshots
    int evalMaxSnapshots = 0;      // 0 = keep all
    bool evalAsync = true;
    int pimcEvery = 500;           // eval vs PIMC every N updates (0 = off)
    int pimcDecks = 60;            // decks per PIMC eval (3 games per deck)
    int pimcWorlds = 16;
    int pimcIters = 400;
    int pimcThreads = 4;
    std::string pimcCsv;           // empty = <out>/eval_vs_pimc.csv
    std::string pimcSaveDir;       // empty = <out>/pimc_snapshots
    int pimcMaxSnapshots = 0;
};

const char* argValue(int argc, char** argv, const char* key, const char* def) {
    std::string k = key;
    for (int i = 1; i + 1 < argc; ++i)
        if (k == argv[i]) return argv[i + 1];
    return def;
}
bool hasFlag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (key == std::string(argv[i])) return true;
    return false;
}

void parseArgs(int argc, char** argv, Args& a) {
    a.updates = std::atoi(argValue(argc, argv, "--updates", std::to_string(a.updates).c_str()));
    a.startUpdate = std::atoi(argValue(argc, argv, "--start-update", "1"));
    a.games = std::atoi(argValue(argc, argv, "--games", std::to_string(a.games).c_str()));
    a.threads = std::atoi(argValue(argc, argv, "--threads",
                                   std::to_string(a.threads).c_str()));
    a.hidden = std::atoi(argValue(argc, argv, "--hidden", std::to_string(a.hidden).c_str()));
    a.lstmHidden = std::atoi(argValue(argc, argv, "--lstm-hidden", std::to_string(a.lstmHidden).c_str()));
    a.epochs = std::atoi(argValue(argc, argv, "--epochs", std::to_string(a.epochs).c_str()));
    a.minibatch = std::atoi(argValue(argc, argv, "--mb", std::to_string(a.minibatch).c_str()));
    a.snapshotEvery = std::atoi(argValue(argc, argv, "--snapshot-every", std::to_string(a.snapshotEvery).c_str()));
    a.lr = float(std::atof(argValue(argc, argv, "--lr", "3e-4")));
    a.clip = float(std::atof(argValue(argc, argv, "--clip", "0.2")));
    a.ent = float(std::atof(argValue(argc, argv, "--ent", "0.03")));
    a.gamma = float(std::atof(argValue(argc, argv, "--gamma", "1.0")));
    a.lambda = float(std::atof(argValue(argc, argv, "--lambda", "0.95")));
    a.targetKL = float(std::atof(argValue(argc, argv, "--target-kl", "0.03")));
    if (hasFlag(argc, argv, "--oracle-reward")) a.oracleReward = true;
    if (hasFlag(argc, argv, "--no-oracle-reward")) a.oracleReward = false;
    a.oracleCoef = float(std::atof(argValue(argc, argv, "--oracle-coef", "0.2")));
    a.poolSize = std::atoi(argValue(argc, argv, "--pool-size", "16"));
    a.archiveSize = std::atoi(argValue(argc, argv, "--archive-size", "16"));
    a.poolEvery = std::atoi(argValue(argc, argv, "--pool-every", "20"));
    a.histProb = float(std::atof(argValue(argc, argv, "--historical-prob", "0.2")));
    a.ruleOppProb = float(std::atof(argValue(argc, argv, "--rule-opponent-prob", "0.15")));
    a.seed = std::atoll(argValue(argc, argv, "--seed", "1"));
    a.out = argValue(argc, argv, "--out", a.out.c_str());
    a.backend = argValue(argc, argv, "--backend", a.backend.c_str());
    const char* res = argValue(argc, argv, "--resume", "");
    if (*res) a.resume = res;
    a.evalEvery = std::atoi(argValue(argc, argv, "--eval-every", std::to_string(a.evalEvery).c_str()));
    if (hasFlag(argc, argv, "--no-eval")) a.evalEvery = 0;
    a.evalDecks = std::atoi(argValue(argc, argv, "--eval-decks", std::to_string(a.evalDecks).c_str()));
    a.evalCsv = argValue(argc, argv, "--eval-csv", "");
    a.evalSaveDir = argValue(argc, argv, "--eval-save-dir", "");
    a.evalMaxSnapshots = std::atoi(argValue(argc, argv, "--eval-max-snapshots", "0"));
    if (hasFlag(argc, argv, "--no-eval-async") || hasFlag(argc, argv, "--eval-sync")) {
        a.evalAsync = false;
    }
    a.pimcEvery = std::atoi(argValue(argc, argv, "--pimc-every", std::to_string(a.pimcEvery).c_str()));
    if (hasFlag(argc, argv, "--no-pimc-eval")) a.pimcEvery = 0;
    a.pimcDecks = std::atoi(argValue(argc, argv, "--pimc-decks", std::to_string(a.pimcDecks).c_str()));
    a.pimcWorlds = std::atoi(argValue(argc, argv, "--pimc-worlds", std::to_string(a.pimcWorlds).c_str()));
    a.pimcIters = std::atoi(argValue(argc, argv, "--pimc-iters", std::to_string(a.pimcIters).c_str()));
    a.pimcThreads = std::atoi(argValue(argc, argv, "--pimc-threads", std::to_string(a.pimcThreads).c_str()));
    a.pimcCsv = argValue(argc, argv, "--pimc-csv", "");
    a.pimcSaveDir = argValue(argc, argv, "--pimc-save-dir", "");
    a.pimcMaxSnapshots = std::atoi(argValue(argc, argv, "--pimc-max-snapshots", "0"));
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    parseArgs(argc, argv, args);
    if (args.threads <= 0) args.threads = 1;

    torch::Device torchDevice = torch::kCPU;
    std::string deviceName = "CPU";
    if (args.backend == "cuda") {
        if (!torch::cuda::is_available()) {
            std::fprintf(stderr,
                         "Backend 'cuda' requested but CUDA is not available in this LibTorch build. "
                         "Pass --backend mps or --backend cpu explicitly.\n");
            return 1;
        }
        torchDevice = torch::kCUDA;
        deviceName = "GPU (CUDA)";
    } else if (args.backend == "mps") {
#if defined(__APPLE__) && (TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
        if (!torch::mps::is_available()) {
            std::fprintf(stderr,
                         "Backend 'mps' requested but MPS is not available (requires macOS 12.3+ on "
                         "Apple Silicon with an MPS-enabled LibTorch). Pass --backend cpu explicitly.\n");
            return 1;
        }
        torchDevice = torch::kMPS;
        deviceName = "GPU (MPS)";
#else
        std::fprintf(stderr,
                     "Backend 'mps' requested but MPS is not supported on this platform/LibTorch build.\n");
        return 1;
#endif
    } else if (args.backend == "cpu") {
        torchDevice = torch::kCPU;
        deviceName = "CPU";
    } else {
        std::fprintf(stderr,
                     "Unknown or unsupported backend '%s'. Valid values: cuda | mps | cpu.\n",
                     args.backend.c_str());
        return 1;
    }

    std::cout << "PerfectPao CPP training (Paodekuai, symmetric single-policy PPO)\n";
    std::cout << "  Compute Device: " << deviceName << "\n";
    if (!args.resume.empty() && !hasFlag(argc, argv, "--start-update")) {
        std::string metaPath = args.resume + "/meta.txt";
        std::ifstream metaFile(metaPath);
        if (metaFile.is_open()) {
            std::string key;
            int val = 0;
            while (metaFile >> key >> val) {
                if (key == "update" || key == "updates") {
                    args.startUpdate = val + 1;
                }
            }
        }
    }

    std::cout << "  updates=" << args.updates
              << (args.startUpdate > 1 ? " (start=" + std::to_string(args.startUpdate) + ")" : "")
              << " games/chunk=" << args.games
              << " threads=" << args.threads
              << " mb=" << args.minibatch
              << " epochs=" << args.epochs
              << " hidden=" << args.hidden
              << " lstm=" << args.lstmHidden
              << " lr=" << args.lr
              << " clip=" << args.clip
              << " ent=" << args.ent
              << " gae(gamma=" << args.gamma << ",lambda=" << args.lambda << ")"
              << " oracle=" << (args.oracleReward ? ("yes(coef=" + std::to_string(args.oracleCoef) + ")") : "no")
              << " targetKL=" << args.targetKL
              << " league(recent=" << args.poolSize << ",archive=" << args.archiveSize
              << ",every=" << args.poolEvery
              << ",hist=" << args.histProb
              << ",rule=" << args.ruleOppProb << ")"
              << " seed=" << args.seed << std::endl;

    if (args.evalEvery > 0) {
        if (args.evalCsv.empty()) {
            args.evalCsv = args.out + "/eval_vs_rule.csv";
        }
        if (args.evalSaveDir.empty()) {
            args.evalSaveDir = args.out + "/eval_snapshots";
        }
        std::cout << "  eval vs RuleAgent: every=" << args.evalEvery
                  << " decks=" << args.evalDecks << " (" << (args.evalDecks * 3) << " games)"
                  << " csv=" << args.evalCsv
                  << " snapshots=" << args.evalSaveDir
                  << " mode=" << (args.evalAsync ? "async" : "sync");
        if (args.evalMaxSnapshots > 0) {
            std::cout << " (keep max " << args.evalMaxSnapshots << ")";
        }
        std::cout << std::endl;
    }
    if (args.pimcEvery > 0) {
        if (args.pimcCsv.empty()) {
            args.pimcCsv = args.out + "/eval_vs_pimc.csv";
        }
        if (args.pimcSaveDir.empty()) {
            args.pimcSaveDir = args.out + "/pimc_snapshots";
        }
        std::cout << "  eval vs PIMC(" << args.pimcWorlds << "x" << args.pimcIters
                  << "): every=" << args.pimcEvery
                  << " decks=" << args.pimcDecks << " (" << (args.pimcDecks * 3) << " games)"
                  << " threads=" << args.pimcThreads
                  << " csv=" << args.pimcCsv
                  << " snapshots=" << args.pimcSaveDir << std::endl;
    }

    std::thread evalThread;
    std::thread pimcEvalThread;

    torch_nn::ModelConfig torchModelCfg;
    torchModelCfg.hidden = args.hidden;
    torchModelCfg.lstmHidden = args.lstmHidden;

    torch_nn::PPOConfig torchPpoCfg;
    torchPpoCfg.lr = args.lr;
    torchPpoCfg.clip = args.clip;
    torchPpoCfg.entCoef = args.ent;
    torchPpoCfg.epochs = args.epochs;
    torchPpoCfg.minibatch = args.minibatch;
    torchPpoCfg.targetKL = args.targetKL;
    torchPpoCfg.criticMaxGradNorm = 5.0f;

    torch_nn::Actor actor(torchModelCfg);
    torch_nn::Critic critic(torchModelCfg);

    // Dedicated CPU instances for multithreaded rollout simulation
    torch_nn::Actor cpuActor(torchModelCfg);
    torch_nn::Critic cpuCritic(torchModelCfg);

    auto syncWeightsToCpu = [&]() {
        torch::NoGradGuard no_grad;
        auto sa = actor->parameters();
        auto da = cpuActor->parameters();
        for (size_t i = 0; i < sa.size(); ++i) da[i].copy_(sa[i]);
        auto sc = critic->parameters();
        auto dc = cpuCritic->parameters();
        for (size_t i = 0; i < sc.size(); ++i) dc[i].copy_(sc[i]);
        cpuActor->eval();
        cpuCritic->eval();
    };

    if (!args.resume.empty()) {
        std::string actPath = args.resume + "/actor.pt";
        std::string critPath = args.resume + "/critic.pt";
        if (std::filesystem::exists(actPath)) actor->load(actPath);
        if (std::filesystem::exists(critPath)) critic->load(critPath);
    }

    auto learner = std::make_unique<torch_nn::PPOLearner>(actor, critic, torchPpoCfg, torchDevice);
    if (!args.resume.empty()) {
        std::string optPath = args.resume + "/opt.pt";
        if (std::filesystem::exists(optPath)) {
            learner->loadOptimizer(optPath);
        }
    }

    syncWeightsToCpu();

    std::shared_ptr<algo::HistoricalPool> histPool = std::make_shared<algo::HistoricalPool>();

    auto pushHistPool = [&](torch_nn::Actor& srcActor, int currentUpdate) {
        auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
        snap->update = currentUpdate;
        snap->actor = torch_nn::Actor(torchModelCfg);
        {
            torch::NoGradGuard no_grad;
            auto srcParams = srcActor->parameters();
            auto dstParams = snap->actor->parameters();
            for (size_t i = 0; i < srcParams.size(); ++i) {
                dstParams[i].copy_(srcParams[i]);
            }
            snap->actor->eval();
        }
        int k = args.poolEvery > 0 ? (currentUpdate / args.poolEvery) : 0;
        bool isPowerOfTwo = (k > 0) && ((k & (k - 1)) == 0);
        bool isCentennial = (currentUpdate % 100 == 0);
        if (args.archiveSize > 0 && (isPowerOfTwo || isCentennial || histPool->archive.empty())) {
            while ((int)histPool->archive.size() >= args.archiveSize && !histPool->archive.empty()) {
                if (histPool->archive.size() > 2) {
                    histPool->archive.erase(histPool->archive.begin() + 1);
                } else {
                    histPool->archive.erase(histPool->archive.begin());
                }
            }
            histPool->archive.push_back(snap);
        }

        if (args.poolSize > 0) {
            while ((int)histPool->recent.size() >= args.poolSize && !histPool->recent.empty()) {
                histPool->recent.erase(histPool->recent.begin());
            }
            histPool->recent.push_back(std::move(snap));
        }
    };

    auto saveHistoricalPool = [&](const std::string& dir) {
        if (!histPool || histPool->empty()) return;
        std::string poolDir = dir + "/hist_pool";
        std::filesystem::create_directories(poolDir);
        std::ofstream meta(poolDir + "/pool_meta.txt");
        if (!meta.is_open()) return;

        meta << "recent_count " << histPool->recent.size() << "\n";
        for (size_t i = 0; i < histPool->recent.size(); ++i) {
            const auto& snap = histPool->recent[i];
            meta << "recent " << i << " " << snap->update << "\n";
            snap->actor->save(poolDir + "/recent_" + std::to_string(i) + "_actor.pt");
        }

        meta << "archive_count " << histPool->archive.size() << "\n";
        for (size_t i = 0; i < histPool->archive.size(); ++i) {
            const auto& snap = histPool->archive[i];
            meta << "archive " << i << " " << snap->update << "\n";
            snap->actor->save(poolDir + "/archive_" + std::to_string(i) + "_actor.pt");
        }
    };

    auto loadHistoricalPool = [&](const std::string& dir) {
        std::string metaPath = dir + "/hist_pool/pool_meta.txt";
        std::ifstream meta(metaPath);
        if (!meta.is_open()) return;

        std::string tag;
        while (meta >> tag) {
            if (tag == "recent" || tag == "archive") {
                size_t idx = 0;
                int snapUpd = 0;
                meta >> idx >> snapUpd;
                auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
                snap->update = snapUpd;
                snap->actor = torch_nn::Actor(torchModelCfg);
                std::string actPath = dir + "/hist_pool/" + tag + "_" +
                                      std::to_string(idx) + "_actor.pt";
                if (std::filesystem::exists(actPath)) {
                    snap->actor->load(actPath);
                }
                snap->actor->eval();
                if (tag == "recent") {
                    histPool->recent.push_back(snap);
                } else {
                    histPool->archive.push_back(snap);
                }
            } else if (tag == "recent_count" || tag == "archive_count") {
                size_t cnt = 0;
                meta >> cnt;
            } else {
                int dummy = 0;
                meta >> dummy;
            }
        }
        std::cout << "  Restored historical pool: recent=" << histPool->recent.size()
                  << " archive=" << histPool->archive.size() << "\n";
    };

    if (!args.resume.empty()) {
        loadHistoricalPool(args.resume);
    }

    auto saveAll = [&](const std::string& dir, int currentUpd = 0) {
        std::filesystem::create_directories(dir);
        actor->save(dir + "/actor.pt");
        critic->save(dir + "/critic.pt");
        learner->saveOptimizer(dir + "/opt.pt");
        if (currentUpd > 0) {
            std::ofstream metaFile(dir + "/meta.txt");
            if (metaFile.is_open()) {
                metaFile << "update " << currentUpd << "\n"
                         << "seed " << args.seed << "\n"
                         << "lr " << args.lr << "\n"
                         << "ent " << args.ent << "\n"
                         << "oracle_reward " << (args.oracleReward ? 1 : 0) << "\n"
                         << "oracle_coef " << args.oracleCoef << "\n";
            }
            saveHistoricalPool(dir);
        }
    };

    std::filesystem::create_directories(args.out);
    saveAll(args.out, 0);

    // warm up the shared oracle DP table before spawning workers
    pdk::minSteps(pdk::CardSet::parse("34567"));

#if defined(_WIN32)
    REASON_CONTEXT awakeReason = {};
    awakeReason.Version = POWER_REQUEST_CONTEXT_VERSION;
    awakeReason.Flags = POWER_REQUEST_CONTEXT_SIMPLE_STRING;
    awakeReason.Reason.SimpleReasonString =
        const_cast<wchar_t*>(L"PerfectPao training");
    HANDLE awake = PowerCreateRequest(&awakeReason);
    if (awake) {
        PowerSetRequest(awake, PowerRequestSystemRequired);
        PowerSetRequest(awake, PowerRequestExecutionRequired);
    }
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED |
                            ES_AWAYMODE_REQUIRED);
#endif

    algo::PPOStats ps{};

    for (int upd = args.startUpdate; upd <= args.updates; ++upd) {
        auto wall0 = std::chrono::steady_clock::now();

        // 1. Synchronous CPU self-play rollout with the shared policy
        algo::ModelSet models{&cpuActor, &cpuCritic};
        algo::RolloutConfig rc;
        rc.gamesPerUpdate = args.games;
        rc.threads = args.threads;
        rc.seed = args.seed + uint64_t(upd) * 7919ULL;
        rc.historicalProb = args.histProb;
        rc.ruleOpponentProb = args.ruleOppProb;
        rc.historicalPool = histPool.get();
        rc.oracleReward = args.oracleReward;
        rc.oracleCoef = args.oracleCoef;

        std::array<std::vector<algo::Transition>, 3> streams;
        algo::RolloutStats rs;
        algo::collectRollout(models, rc, streams, rs);

        // 2. Merge the three seat streams (same shared policy) and update
        size_t actualSamples = 0;
        auto tLearn0 = std::chrono::steady_clock::now();
        {
            std::vector<algo::Transition> all;
            size_t total = 0;
            for (int s = 0; s < 3; ++s) total += streams[s].size();
            all.reserve(total);
            for (int s = 0; s < 3; ++s) {
                for (auto& t : streams[s]) all.push_back(std::move(t));
                streams[s].clear();
                streams[s].shrink_to_fit();
            }
            actualSamples = all.size();
            if (!all.empty()) {
                try {
                    auto batchTensors = torch_nn::transitionsToBatch(
                        all, args.gamma, args.lambda, torchDevice);
                    learner->setLr(args.lr);
                    learner->setEntCoef(args.ent);
                    auto torchStats = learner->update(batchTensors);
                    ps.epochsCompleted = torchStats.epochsCompleted;
                    ps.entropy = torchStats.entropy;
                    ps.lastEpochKL = torchStats.kl;
                    ps.clipFraction = torchStats.clipFraction;
                    ps.pgLoss = torchStats.pgLoss;
                    ps.vLoss = torchStats.vLoss;
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "Update error: %s\n", e.what());
                }
            }
        }
        double learnSecs = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - tLearn0)
                               .count();

        // 3. Sync freshly trained weights back to the CPU rollout models
        syncWeightsToCpu();

        double wallSecs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - wall0)
                              .count();

        double bpg = double(rs.bombs) / std::max(1, rs.games);
        double mpg = double(rs.moves) / std::max(1, rs.games);
        double margin = rs.winnerMargin / std::max(1, rs.games);
        std::string pureStr;
        if (rs.pureSelfPlayGames > 0) {
            char pbuf[64];
            std::snprintf(pbuf, sizeof(pbuf), " (pure margin %.3f [%d g])",
                          rs.pureSelfPlayMargin / rs.pureSelfPlayGames,
                          rs.pureSelfPlayGames);
            pureStr = pbuf;
        }

        std::printf(
            "upd %4d | wall %.1fs learn %.2fs games %d mb %d ep %d/%d | "
            "wins %lld/%lld/%lld margin %6.3f%s bomb/g %.2f moves/g %.1f | "
            "lr %.2e ent %5.3f vL %7.2f kl %.4f cf %.2f | train_n %zu\n",
            upd, wallSecs, learnSecs,
            rs.games, args.minibatch,
            ps.epochsCompleted, args.epochs,
            rs.wins[0], rs.wins[1], rs.wins[2], margin, pureStr.c_str(), bpg, mpg,
            args.lr, ps.entropy, ps.vLoss, ps.lastEpochKL, ps.clipFraction,
            actualSamples);
        std::fflush(stdout);

        if (args.poolEvery > 0 && upd % args.poolEvery == 0) {
            pushHistPool(cpuActor, upd);
        }
        if (args.snapshotEvery > 0 &&
            (upd % args.snapshotEvery == 0 || upd == args.updates)) {
            saveAll(args.out, upd);
        }

        if (args.evalEvery > 0 &&
            (upd % args.evalEvery == 0 || upd == args.updates)) {
            if (evalThread.joinable()) {
                evalThread.join();
            }
            auto cloned = algo::cloneActor(cpuActor);
            algo::RuleEvalConfig evalCfg;
            evalCfg.decks = args.evalDecks;
            evalCfg.seed = static_cast<uint64_t>(upd * 10007 + 1);
            evalCfg.csvPath = args.evalCsv;
            evalCfg.label = "u" + std::to_string(upd);
            evalCfg.saveDir = args.evalSaveDir + "/" + evalCfg.label;
            evalCfg.maxSnapshots = args.evalMaxSnapshots;
            evalCfg.update = upd;
            evalCfg.elapsedMinutes = wallSecs / 60.0;
            evalCfg.verbose = false;

            auto runEval = [cloned = std::move(cloned), evalCfg]() {
                auto res = algo::evaluateAgainstRule(cloned, evalCfg);
                if (res.ok) {
                    std::printf("[eval %s] vs RuleAgent games %d | WP %.3f ADP %+.3f\n",
                                evalCfg.label.c_str(), res.games, res.wp, res.adp);
                } else {
                    std::printf("[eval %s] skipped: %s\n", evalCfg.label.c_str(), res.error.c_str());
                }
                std::fflush(stdout);
            };

            if (args.evalAsync) {
                evalThread = std::thread(std::move(runEval));
            } else {
                runEval();
            }
        }

        if (args.pimcEvery > 0 &&
            (upd % args.pimcEvery == 0 || upd == args.updates)) {
            if (pimcEvalThread.joinable()) {
                pimcEvalThread.join();
            }
            auto pimcCloned = algo::cloneActor(cpuActor);
            algo::PimcEvalConfig pcfg;
            pcfg.decks = args.pimcDecks;
            pcfg.seed = static_cast<uint64_t>(upd * 60013 + 7);
            pcfg.pimc.worlds = args.pimcWorlds;
            pcfg.pimc.iterations = args.pimcIters;
            pcfg.threads = args.pimcThreads;
            pcfg.csvPath = args.pimcCsv;
            pcfg.label = "u" + std::to_string(upd);
            pcfg.saveDir = args.pimcSaveDir + "/" + pcfg.label;
            pcfg.maxSnapshots = args.pimcMaxSnapshots;
            pcfg.update = upd;
            pcfg.elapsedMinutes = wallSecs / 60.0;

            auto runPimcEval = [pimcCloned = std::move(pimcCloned), pcfg]() {
                auto res = algo::evaluateAgainstPimc(pimcCloned, pcfg);
                if (res.ok) {
                    std::printf("[pimc %s] %dx%d games %d | WP %.3f ADP %+.3f\n",
                                pcfg.label.c_str(), pcfg.pimc.worlds,
                                pcfg.pimc.iterations, res.games, res.wp, res.adp);
                } else {
                    std::printf("[pimc %s] skipped: %s\n", pcfg.label.c_str(), res.error.c_str());
                }
                std::fflush(stdout);
            };

            if (args.evalAsync) {
                pimcEvalThread = std::thread(std::move(runPimcEval));
            } else {
                runPimcEval();
            }
        }
    }

    if (evalThread.joinable()) evalThread.join();
    if (pimcEvalThread.joinable()) pimcEvalThread.join();

#if defined(_WIN32)
    if (awake) {
        PowerClearRequest(awake, PowerRequestExecutionRequired);
        PowerClearRequest(awake, PowerRequestSystemRequired);
        CloseHandle(awake);
    }
    SetThreadExecutionState(ES_CONTINUOUS);
#endif

    saveAll(args.out, args.updates);
    std::cout << "models saved to " << args.out << "/\n";
    return 0;
}
