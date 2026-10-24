#include "algo/eval_pimc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

#include "algo/eval_common.h"
#include "pdk/features.h"
#include "pdk/game.h"
#include "pdk/oracle.h"

namespace algo {

namespace {

// Plays one game: `policySeat` is the model, the other two seats are PIMC AIs.
// Returns {won, payoff} from the policy seat's perspective.
std::pair<bool, double> playOneGame(torch_nn::Actor& actor, std::mutex& actorMu,
                                    const pdk::Game& proto, int policySeat,
                                    const pdk::PimcConfig& pimcCfg,
                                    uint64_t gameSeed) {
    pdk::Game g = proto;
    pdk::PimcAi ai(pimcCfg);
    ai.reseed(gameSeed);
    torch::NoGradGuard no_grad;

    std::vector<pdk::LegalOption> opts;
    int guard = 0;
    while (!g.over && guard < 10000) {
        if (g.turn == policySeat) {
            pdk::legalOptions(g, opts);
            if (opts.empty()) break;
            // single-sample forward, serialized across worker threads
            std::lock_guard<std::mutex> lk(actorMu);
            torch::Tensor xImpT = torch::empty({1, torch_nn::kImpInput}, torch::kFloat32);
            torch::Tensor seqT = torch::empty({1, torch_nn::kLstmSteps, torch_nn::kLstmIn}, torch::kFloat32);
            int64_t L = static_cast<int64_t>(opts.size());
            torch::Tensor actFeatT = torch::empty({L, torch_nn::kActionInput}, torch::kFloat32);
            torch::Tensor actSampleT = torch::zeros({L}, torch::kLong);
            torch::Tensor actIdT = torch::empty({L}, torch::kLong);

            pdk::EncodedState e = pdk::encodeState(g);
            float* pXImp = xImpT.data_ptr<float>();
            for (int j = 0; j < pdk::kNodeBin; ++j)
                pXImp[j] = float(e.imp[static_cast<size_t>(j)]);
            std::copy(e.scalar.begin(), e.scalar.end(), pXImp + pdk::kNodeBin);
            float* pSeq = seqT.data_ptr<float>();
            for (int k = 0; k < pdk::kHistoryLen; ++k) {
                const uint8_t* src = e.imp.data() + (pdk::kStaticMatrices + k) * pdk::kCardMat;
                float* dst = pSeq + k * pdk::kCardMat;
                for (int j = 0; j < pdk::kCardMat; ++j) dst[j] = float(src[j]);
            }
            float* pActFeat = actFeatT.data_ptr<float>();
            int64_t* pActId = actIdT.data_ptr<int64_t>();
            for (size_t i = 0; i < opts.size(); ++i) {
                std::memcpy(pActFeat + int64_t(i) * torch_nn::kActionInput,
                            opts[i].feature.data(), torch_nn::kActionInput * sizeof(float));
                pActId[i] = opts[i].abstractId;
            }

            torch::Tensor logitsT = actor->forward(xImpT, seqT, actFeatT, actSampleT, actIdT);
            const float* row = logitsT.data_ptr<float>();
            const pdk::LegalOption* best = &opts.front();
            float bestVal = -1e30f;
            for (const auto& o : opts) {
                float v = row[o.abstractId];
                if (v > bestVal) {
                    bestVal = v;
                    best = &o;
                }
            }
            g.step(best->concrete);
        } else {
            std::vector<pdk::CardSet> legal = g.legal();
            if (legal.empty()) break;
            g.step(ai.selectMove(g, g.turn));
        }
        ++guard;
    }
    return {g.winner == policySeat, g.payoff(policySeat)};
}

}  // namespace

PimcEvalResult evaluateAgainstPimc(torch_nn::Actor& actor,
                                   const PimcEvalConfig& cfg) {
    PimcEvalResult res;
    if (cfg.decks <= 0) {
        res.error = "deck count must be positive";
        return res;
    }

    uint64_t seed = cfg.seed;
    if (seed == 0) {
        seed = static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    }
    pdk::Rng rng(seed);

    // Pre-deal every deck (deterministic), then play 3 seat-rotated games.
    const int totalGames = cfg.decks * 3;
    std::vector<pdk::Game> protos;
    protos.reserve(static_cast<size_t>(cfg.decks));
    for (int d = 0; d < cfg.decks; ++d) {
        pdk::Game g;
        g.deal(rng);
        protos.push_back(g);
    }

    std::atomic<int> nextGame{0};
    std::atomic<int> wins{0};
    std::atomic<int> played{0};
    std::mutex scoreMu;
    double score = 0.0;
    std::mutex actorMu;

    int threads = std::max(1, cfg.threads);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            pdk::clearOracleCache();  // thread-local memo released at exit
            for (;;) {
                int gi = nextGame.fetch_add(1);
                if (gi >= totalGames) break;
                int deckIdx = gi / 3;
                int policySeat = gi % 3;
                uint64_t gameSeed = seed + uint64_t(gi) * 7919ULL + 13ULL;
                auto [won, payoff] = playOneGame(actor, actorMu, protos[static_cast<size_t>(deckIdx)],
                                                 policySeat, cfg.pimc, gameSeed);
                if (won) wins.fetch_add(1);
                played.fetch_add(1);
                {
                    std::lock_guard<std::mutex> lk(scoreMu);
                    score += payoff;
                }
            }
            pdk::clearOracleCache();
        });
    }
    for (auto& w : workers) w.join();

    res.decks = cfg.decks;
    res.games = played.load();
    if (res.games > 0) {
        res.wp = double(wins.load()) / res.games;
        res.adp = score / res.games;
        res.ok = (res.games == totalGames);
    } else {
        res.error = "no games completed";
        return res;
    }

    if (res.ok && !cfg.csvPath.empty()) {
        char row[256];
        std::snprintf(row, sizeof(row),
                      "%d,%.2f,%s,%s,%d,%d,%.4f,%.4f,%d,%d",
                      cfg.update, cfg.elapsedMinutes,
                      eval_common::timestampNow().c_str(),
                      (cfg.label.empty() ? ("u" + std::to_string(cfg.update)) : cfg.label).c_str(),
                      res.decks, res.games, res.wp, res.adp,
                      cfg.pimc.worlds, cfg.pimc.iterations);
        eval_common::appendCsv(cfg.csvPath,
                               "update,minutes,timestamp,label,decks,games,wp,adp,worlds,iterations",
                               row);
    }

    if (res.ok && !cfg.saveDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cfg.saveDir, ec);
        actor->save(cfg.saveDir + "/actor.pt");
        if (FILE* mf = std::fopen((cfg.saveDir + "/meta.txt").c_str(), "a")) {
            std::fprintf(mf, "update %d\ngames %d\nwp %.4f\nadp %.4f\nworlds %d\niterations %d\n",
                         cfg.update, res.games, res.wp, res.adp,
                         cfg.pimc.worlds, cfg.pimc.iterations);
            std::fclose(mf);
        }
        std::filesystem::path saveRoot(cfg.saveDir);
        if (saveRoot.has_parent_path()) {
            eval_common::pruneSnapshots(saveRoot.parent_path().string(), cfg.maxSnapshots);
        }
    }

    return res;
}

PimcEvalResult evaluateAgainstPimc(std::shared_ptr<torch_nn::Actor> actor,
                                   const PimcEvalConfig& cfg) {
    return evaluateAgainstPimc(*actor, cfg);
}

}  // namespace algo
