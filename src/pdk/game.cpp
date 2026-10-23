#include "pdk/game.h"

#include <algorithm>

namespace pdk {

void Game::deal(Rng& rng) {
    *this = Game{};
    auto deck = shuffledDeck(rng);
    for (int s = 0; s < 3; ++s)
        for (int i = s * kHandSize; i < (s + 1) * kHandSize; ++i)
            hand[s].add(deck[i]);
    // Designate one of the four 3s uniformly; its holder leads (equivalent to
    // picking a player weighted by how many 3s they hold).
    int pick = rng.below(4);
    int acc = 0;
    for (int s = 0; s < 3; ++s) {
        acc += hand[s].c[0];
        if (pick < acc) {
            turn = s;
            break;
        }
    }
    lastPlayer = turn;
}

CardSet Game::toBeat() const {
    if (seq.empty()) return CardSet{};
    if (!seq.back().empty()) return seq.back();
    // The last action was a pass: either two consecutive passes (free lead)
    // or the action before the pass must be beaten.
    if (seq.size() >= 2 && seq[seq.size() - 2].empty()) return CardSet{};
    if (seq.size() >= 2) return seq[seq.size() - 2];
    return CardSet{};
}

std::vector<CardSet> Game::legal() const {
    if (over) return {};
    CardSet tb = toBeat();
    bool firstMove = seq.empty();
    if (tb.empty()) {
        // free lead: any combination, pass never legal
        std::vector<CardSet> ms = genAllMoves(hand[turn]);
        if (firstMove) {
            // the first hand must contain a 3
            size_t w = 0;
            for (auto& m : ms)
                if (m.c[0] >= 1) ms[w++] = std::move(m);
            ms.resize(w);
            return ms;
        }
        if (reported(nextSeat())) {
            // leading a solo while the next player reported: must top with
            // the largest solo in hand
            int largest = -1;
            for (int r = 0; r < kRanks; ++r)
                if (hand[turn].c[r] >= 1) largest = r;
            size_t w = 0;
            for (auto& m : ms) {
                MoveInfo info = detectMove(m);
                if (info.type != MT_SINGLE || info.rank == largest)
                    ms[w++] = std::move(m);
            }
            ms.resize(w);
            return ms;
        }
        return ms;
    }
    return legalMoves(hand[turn], tb);  // must-beat semantics inside
}

void Game::step(const CardSet& m) {
    if (over) return;
    if (!m.empty()) {
        lastPlayer = turn;
        if (detectMove(m).type == MT_BOMB) ++bombCount;
        hand[turn].sub(m);
        for (int r = 0; r < kRanks; ++r) played[turn].add(r, m.c[r]);
        if (hand[turn].total() == 0) {
            over = true;
            winner = turn;
        }
    }
    lastMove[turn] = m;
    seq.push_back(m);
    if (!over) turn = (turn + 1) % 3;
}

double Game::payoff(int seat) const {
    if (!over) return 0.0;
    auto payment = [&](int s) -> double {
        if (s == winner) return 0.0;
        int left = hand[s].total();
        int mult = (left == kHandSize) ? 2 : 1;  // never played a card: double
        return double(left * mult);
    };
    if (seat == winner) {
        double sum = 0.0;
        for (int s = 0; s < 3; ++s) sum += payment(s);
        return sum / kHandSize;
    }
    return -payment(seat) / kHandSize;
}

}  // namespace pdk
