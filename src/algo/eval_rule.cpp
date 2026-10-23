#include "algo/eval_rule.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <vector>

#include "pdk/features.h"
#include "pdk/game.h"
#include "pdk/rule_agent.h"

namespace algo {

std::shared_ptr<torch_nn::Actor> cloneActor(torch_nn::Actor src) {
    auto dst = std::make_shared<torch_nn::Actor>(src->config);
    torch::NoGradGuard no_grad;
    auto srcParams = src->parameters();
    auto dstParams = (*dst)->parameters();
    for (size_t i = 0; i < srcParams.size(); ++i) {
        dstParams[i].copy_(srcParams[i]);
    }
    (*dst)->eval();
    return dst;
}

namespace {

// Plays all evaluation games in lockstep: rule-agent seats act inline, the
// policy seat is batched across games for throughput.
struct EvalGame {
    pdk::Game g;
    int policySeat = 0;
    bool done = false;
};

void pruneSnapshots(const std::string& saveRoot, int maxSnapshots) {
    if (maxSnapshots <= 0 || saveRoot.empty()) return;
    std::error_code ec;
    if (!std::filesystem::exists(saveRoot, ec)) return;
    std::vector<std::pair<long long, std::filesystem::path>> dirs;
    for (const auto& entry : std::filesystem::directory_iterator(saveRoot, ec)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() < 2 || name[0] != 'u') continue;
        long long upd = 0;
        try {
            upd = std::stoll(name.substr(1));
        } catch (...) {
            continue;
        }
        dirs.emplace_back(upd, entry.path());
    }
    if (static_cast<int>(dirs.size()) > maxSnapshots) {
        std::sort(dirs.begin(), dirs.end());
        int toRemove = static_cast<int>(dirs.size()) - maxSnapshots;
        for (int i = 0; i < toRemove; ++i) {
            std::filesystem::remove_all(dirs[i].second, ec);
        }
    }
}

RuleEvalResult runEval(torch_nn::Actor& actor, const RuleEvalConfig& cfg) {
    RuleEvalResult res;
    if (cfg.decks <= 0) {
        res.error = "deck count must be positive";
        return res;
    }

    uint64_t seed = cfg.seed;
    if (seed == 0) {
        seed = static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    }
    pdk::Rng rng(seed);

    std::vector<EvalGame> games;
    games.reserve(cfg.decks * 3);
    for (int d = 0; d < cfg.decks; ++d) {
        pdk::Game proto;
        proto.deal(rng);
        for (int seat = 0; seat < 3; ++seat) {
            EvalGame eg;
            eg.g = proto;
            eg.policySeat = seat;
            games.push_back(std::move(eg));
        }
    }

    torch::NoGradGuard no_grad;
    int done = 0;
    int wins = 0;
    double score = 0.0;

    std::vector<int> batchGi;
    std::vector<std::vector<pdk::LegalOption>> batchOpts;

    while (done < static_cast<int>(games.size())) {
        // rule-agent moves first, inline
        for (size_t gi = 0; gi < games.size(); ++gi) {
            EvalGame& eg = games[gi];
            if (eg.done || eg.g.turn == eg.policySeat) continue;
            std::vector<pdk::CardSet> legal = eg.g.legal();
            pdk::CardSet m = pdk::RuleAgent::selectMove(eg.g, eg.g.turn, legal);
            eg.g.step(m);
            if (eg.g.over) {
                eg.done = true;
                ++done;
                if (eg.g.winner == eg.policySeat) ++wins;
                score += eg.g.payoff(eg.policySeat);
            }
        }

        // policy moves, batched
        batchGi.clear();
        batchOpts.clear();
        for (size_t gi = 0; gi < games.size(); ++gi) {
            EvalGame& eg = games[gi];
            if (eg.done || eg.g.turn != eg.policySeat) continue;
            batchGi.push_back(static_cast<int>(gi));
            batchOpts.emplace_back();
            pdk::legalOptions(eg.g, batchOpts.back());
        }
        if (batchGi.empty()) continue;

        int64_t B = static_cast<int64_t>(batchGi.size());
        int64_t totalOptions = 0;
        for (const auto& os : batchOpts) totalOptions += static_cast<int64_t>(os.size());

        torch::Tensor xImpT = torch::empty({B, torch_nn::kImpInput}, torch::kFloat32);
        torch::Tensor seqT = torch::empty({B, torch_nn::kLstmSteps, torch_nn::kLstmIn}, torch::kFloat32);
        torch::Tensor actFeatT = torch::empty({totalOptions, torch_nn::kActionInput}, torch::kFloat32);
        torch::Tensor actSampleT = torch::empty({totalOptions}, torch::kLong);
        torch::Tensor actIdT = torch::empty({totalOptions}, torch::kLong);

        float* pXImp = xImpT.data_ptr<float>();
        float* pSeq = seqT.data_ptr<float>();
        float* pActFeat = actFeatT.data_ptr<float>();
        int64_t* pActSample = actSampleT.data_ptr<int64_t>();
        int64_t* pActId = actIdT.data_ptr<int64_t>();

        int64_t actOff = 0;
        for (int64_t i = 0; i < B; ++i) {
            pdk::EncodedState e = pdk::encodeState(games[batchGi[i]].g);
            for (int j = 0; j < pdk::kNodeBin; ++j)
                pXImp[i * torch_nn::kImpInput + j] = float(e.imp[j]);
            std::copy(e.scalar.begin(), e.scalar.end(), pXImp + i * torch_nn::kImpInput + pdk::kNodeBin);
            float* curSeq = pSeq + i * pdk::kHistoryLen * pdk::kCardMat;
            for (int k = 0; k < pdk::kHistoryLen; ++k) {
                const uint8_t* src = e.imp.data() + (pdk::kStaticMatrices + k) * pdk::kCardMat;
                float* dst = curSeq + k * pdk::kCardMat;
                for (int j = 0; j < pdk::kCardMat; ++j) dst[j] = float(src[j]);
            }
            for (const auto& o : batchOpts[i]) {
                std::memcpy(pActFeat + actOff * torch_nn::kActionInput, o.feature.data(),
                            torch_nn::kActionInput * sizeof(float));
                pActSample[actOff] = i;
                pActId[actOff] = o.abstractId;
                ++actOff;
            }
        }

        torch::Tensor logitsT = actor->forward(xImpT, seqT, actFeatT, actSampleT, actIdT);
        const float* pLogits = logitsT.data_ptr<float>();

        for (int64_t i = 0; i < B; ++i) {
            EvalGame& eg = games[batchGi[i]];
            const auto& os = batchOpts[i];
            // greedy argmax over legal options
            const float* row = pLogits + i * torch_nn::kNumActions;
            const pdk::LegalOption* best = &os.front();
            float bestVal = -1e30f;
            for (const auto& o : os) {
                float v = row[o.abstractId];
                if (v > bestVal) {
                    bestVal = v;
                    best = &o;
                }
            }
            eg.g.step(best->concrete);
            if (eg.g.over) {
                eg.done = true;
                ++done;
                if (eg.g.winner == eg.policySeat) ++wins;
                score += eg.g.payoff(eg.policySeat);
            }
        }
    }

    res.decks = cfg.decks;
    res.games = static_cast<int>(games.size());
    res.wp = double(wins) / res.games;
    res.adp = score / res.games;
    res.ok = true;

    if (!cfg.csvPath.empty()) {
        std::filesystem::path p(cfg.csvPath);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
        }
        bool needHeader = true;
        if (FILE* probe = std::fopen(cfg.csvPath.c_str(), "rb")) {
            std::fseek(probe, 0, SEEK_END);
            needHeader = (std::ftell(probe) <= 0);
            std::fclose(probe);
        }
        FILE* f = std::fopen(cfg.csvPath.c_str(), "ab");
        if (f) {
            if (needHeader) {
                std::fprintf(f, "update,minutes,timestamp,label,decks,games,wp,adp\n");
            }
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            char timeBuf[32];
            std::tm tmNow{};
#if defined(_WIN32)
            localtime_s(&tmNow, &t);
#else
            localtime_r(&t, &tmNow);
#endif
            std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &tmNow);
            std::string label = cfg.label.empty() ? ("u" + std::to_string(cfg.update)) : cfg.label;
            std::fprintf(f, "%d,%.2f,%s,%s,%d,%d,%.4f,%.4f\n",
                         cfg.update, cfg.elapsedMinutes, timeBuf, label.c_str(),
                         res.decks, res.games, res.wp, res.adp);
            std::fclose(f);
        }
    }

    if (!cfg.saveDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cfg.saveDir, ec);
        actor->save(cfg.saveDir + "/actor.pt");
        if (FILE* mf = std::fopen((cfg.saveDir + "/meta.txt").c_str(), "a")) {
            std::fprintf(mf, "update %d\ngames %d\nwp %.4f\nadp %.4f\n",
                         cfg.update, res.games, res.wp, res.adp);
            std::fclose(mf);
        }
        std::filesystem::path saveRoot(cfg.saveDir);
        if (saveRoot.has_parent_path()) {
            pruneSnapshots(saveRoot.parent_path().string(), cfg.maxSnapshots);
        }
    }

    return res;
}

}  // namespace

RuleEvalResult evaluateAgainstRule(torch_nn::Actor& actor,
                                   const RuleEvalConfig& cfg) {
    return runEval(actor, cfg);
}

RuleEvalResult evaluateAgainstRule(std::shared_ptr<torch_nn::Actor> actor,
                                   const RuleEvalConfig& cfg) {
    return runEval(*actor, cfg);
}

}  // namespace algo
