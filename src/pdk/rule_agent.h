// Lightweight heuristic rule agent for Paodekuai (evaluation anchor baseline).
//   1. Finish the hand immediately when possible.
//   2. Free leading: prefer shedding multi-card combos from low ranks, keep
//      bombs; when the next player reported, the engine already forces solo
//      topping.
//   3. Following: beat with the smallest non-bomb play; bomb only when an
//      opponent is close to going out (or must-beat forces it).
#pragma once

#include <vector>

#include "pdk/cards.h"
#include "pdk/game.h"
#include "pdk/moves.h"

namespace pdk {

class RuleAgent {
public:
    static CardSet selectMove(const Game& g, int seat,
                             const std::vector<CardSet>& legal) {
        if (legal.empty()) return CardSet();
        if (legal.size() == 1) return legal.front();

        // Finish now if any legal move empties the hand.
        for (const CardSet& m : legal) {
            if (!m.empty() && m.total() == g.hand[seat].total()) return m;
        }

        CardSet tb = g.toBeat();
        bool leading = tb.empty() || g.lastPlayer == seat;

        // Split legal moves into bombs / non-bombs / pass.
        std::vector<CardSet> bombs, nonBombs;
        bool hasPass = false;
        for (const CardSet& m : legal) {
            if (m.empty()) { hasPass = true; continue; }
            if (detectMove(m).type == MT_BOMB) bombs.push_back(m);
            else nonBombs.push_back(m);
        }

        // Min enemy hand count (how close a rival is to going out).
        int minRivalHand = kHandSize;
        for (int s = 0; s < 3; ++s)
            if (s != seat) minRivalHand = std::min(minRivalHand, g.hand[s].total());

        auto lowestMainRank = [](const std::vector<CardSet>& v) {
            size_t best = 0;
            int bestScore = 1 << 30;
            for (size_t i = 0; i < v.size(); ++i) {
                MoveInfo info = detectMove(v[i]);
                // shed more cards earlier, from lower ranks
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

        // Following: prefer the smallest non-bomb beat.
        if (!nonBombs.empty()) {
            // Urgency: if the rival who leads is about to go out, prefer the
            // biggest beat instead of the smallest.
            int rivalHand = g.hand[g.lastPlayer].total();
            if (rivalHand <= 2) {
                CardSet best = nonBombs.front();
                for (const CardSet& m : nonBombs) {
                    if (detectMove(m).rank > detectMove(best).rank) best = m;
                }
                return best;
            }
            return lowestMainRank(nonBombs);
        }
        // Only bombs can beat (must-beat may force them).
        if (!bombs.empty()) {
            if (minRivalHand <= 3 || !hasPass) return bombs.front();
        }
        if (hasPass) return CardSet();
        return legal.front();
    }
};

}  // namespace pdk
