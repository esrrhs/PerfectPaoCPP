// Heuristic rule agents for Paodekuai (league opponents / rollout policy /
// evaluation anchors).
//
// Two flavours:
//   RuleAgent::selectMove   — cheap greedy policy (PIMC fast rollout fallback).
//   PlanAgent::selectMove   — strengthened planner built on the Oracle
//                             min-steps solver: picks the legal move that
//                             leaves the hand with the best (lowest) minimum
//                             play-out step count, with tactical guards:
//                             finish-now, bomb preservation, low-rank beats,
//                             holding back the single 2 while leading.
#pragma once

#include <vector>

#include "pdk/cards.h"
#include "pdk/game.h"
#include "pdk/moves.h"
#include "pdk/oracle.h"

namespace pdk {

// Cheap greedy policy: finish now if possible, otherwise shed the lowest-rank
// / largest combo, save bombs, beat as cheap as possible. The core works on
// raw hand arrays so the MCTS rollout can reuse it on simulated states.
class RuleAgent {
public:
    static CardSet greedyChoose(const std::array<CardSet, 3>& hands,
                                const CardSet& toBeat, int lastPlayer, int seat,
                                const std::vector<CardSet>& legal) {
        if (legal.empty()) return CardSet();
        if (legal.size() == 1) return legal.front();

        for (const CardSet& m : legal) {
            if (!m.empty() && m.total() == hands[static_cast<size_t>(seat)].total())
                return m;
        }

        bool leading = toBeat.empty() || lastPlayer == seat;

        std::vector<CardSet> bombs, nonBombs;
        bool hasPass = false;
        for (const CardSet& m : legal) {
            if (m.empty()) { hasPass = true; continue; }
            if (detectMove(m).type == MT_BOMB) bombs.push_back(m);
            else nonBombs.push_back(m);
        }

        int minRivalHand = kHandSize;
        for (int s = 0; s < 3; ++s)
            if (s != seat)
                minRivalHand = std::min(minRivalHand, hands[static_cast<size_t>(s)].total());

        auto lowestMainRank = [](const std::vector<CardSet>& v) {
            size_t best = 0;
            int bestScore = 1 << 30;
            for (size_t i = 0; i < v.size(); ++i) {
                MoveInfo info = detectMove(v[i]);
                int score = info.rank * 16 - v[i].total();
                if (score < bestScore) {
                    bestScore = score;
                    best = i;
                }
            }
            return v[best];
        };

        if (leading) {
            if (!nonBombs.empty()) return lowestMainRank(nonBombs);
            return bombs.empty() ? legal.front() : bombs.front();
        }

        if (!nonBombs.empty()) {
            int rivalHand = hands[static_cast<size_t>(lastPlayer)].total();
            if (rivalHand <= 2) {
                CardSet best = nonBombs.front();
                for (const CardSet& m : nonBombs) {
                    if (detectMove(m).rank > detectMove(best).rank) best = m;
                }
                return best;
            }
            return lowestMainRank(nonBombs);
        }
        if (!bombs.empty()) {
            if (minRivalHand <= 3 || !hasPass) return bombs.front();
        }
        if (hasPass) return CardSet();
        return legal.front();
    }

    static CardSet selectMove(const Game& g, int seat,
                             const std::vector<CardSet>& legal) {
        return greedyChoose(g.hand, g.toBeat(), g.lastPlayer, seat, legal);
    }
};

// Strengthened Oracle-planning agent (deterministic).
class PlanAgent {
public:
    // Larger is better. Score = -minSteps(after) with tactical adjustments.
    static double moveScore(const Game& g, int seat, const CardSet& m,
                            const CardSet& tb) {
        MoveInfo info = detectMove(m);
        CardSet after = g.hand[seat];
        after.sub(m);
        if (after.empty()) return 1e9;  // finish now

        double score = -double(minSteps(after)) * 10.0;
        score -= double(info.rank);  // prefer shedding low ranks

        bool leading = tb.empty() || g.lastPlayer == seat;
        if (leading) {
            score += 0.5 * double(m.total());  // shed more cards per lead
            // keep bombs while leading (they are also defensive must-beat
            // liabilities); breaking even potential bombs is discouraged
            if (info.type == MT_BOMB) score -= 6.0;
            // hold the lone 2 while leading unless it completes the hand
            if (info.type == MT_SINGLE && info.rank == 12) score -= 4.0;
            // if the next player reported, avoid handing them a solo win:
            // engine forces the largest solo already; multi-card leads safe
        } else {
            // following: never break a four-of-a-kind unless the move IS the
            // bomb (the 4th card of a bomb rank spent as kicker is a loss)
            int r = info.rank;
            bool breaksBomb = false;
            for (int rr = 0; rr < kRanks; ++rr) {
                if (g.hand[seat].c[rr] == 4 && m.c[rr] > 0 && m.c[rr] < 4)
                    breaksBomb = true;
            }
            (void)r;
            if (breaksBomb) score -= 8.0;
            if (info.type == MT_BOMB) {
                // bomb only pays when the rival is about to go out or the
                // bomb directly enables a finish
                int rivalHand = g.hand[g.lastPlayer].total();
                bool closesGame = minSteps(after) <= 1;
                if (rivalHand > 2 && !closesGame) score -= 15.0;
            }
            // spending the lone 2 to beat something small is wasteful
            if (info.type == MT_SINGLE && info.rank == 12) {
                MoveInfo rival = detectMove(tb);
                if (rival.rank < 10) score -= 5.0;
            }
        }
        return score;
    }

    static CardSet selectMove(const Game& g, int seat,
                             const std::vector<CardSet>& legal) {
        if (legal.empty()) return CardSet();
        if (legal.size() == 1) return legal.front();
        CardSet tb = g.toBeat();
        CardSet best = legal.front();
        double bestScore = -1e18;
        for (const CardSet& m : legal) {
            double s = moveScore(g, seat, m, tb);
            if (s > bestScore) {
                bestScore = s;
                best = m;
            }
        }
        return best;
    }
};

}  // namespace pdk
