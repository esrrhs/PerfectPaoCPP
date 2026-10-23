// Paodekuai game engine (Hunan 3-player 16-card rules, docs/RULES.md).
//
// All 48 cards are dealt (16 per player). One of the four 3s is designated at
// random and its holder leads; the first hand must contain a 3. Must-beat:
// when following, pass is legal only if nothing beats the table. Report-single
// (报单): a player holding exactly one card is "reported"; when leading a solo
// while the next player is reported, the lead must be the largest solo.
//
// Seats are fully symmetric (every player for themselves). Play order is
// 0 -> 1 -> 2 -> 0.
#pragma once

#include <array>
#include <vector>

#include "pdk/cards.h"
#include "pdk/moves.h"

namespace pdk {

struct Game {
    std::array<CardSet, 3> hand;
    std::array<CardSet, 3> played;
    std::array<CardSet, 3> lastMove;  // last move per seat (empty on pass)
    std::vector<CardSet> seq;         // full action sequence including passes

    int turn = 0;          // acting seat
    int lastPlayer = 0;    // last seat that made a non-pass move
    int bombCount = 0;     // bombs shown so far
    bool over = false;
    int winner = -1;       // seat that emptied its hand

    void deal(Rng& rng);

    // The play that must currently be beaten; empty means free lead.
    CardSet toBeat() const;

    // Legal moves for the acting seat including all engine constraints
    // (first-hand-3, must-beat, report-single topping).
    std::vector<CardSet> legal() const;

    // Applies the move (must be legal); handles pass, bomb counting, win test.
    void step(const CardSet& m);

    // A seat holding exactly its last card has "reported".
    bool reported(int seat) const { return hand[seat].total() == 1; }

    // Zero-sum payoff from the seat's perspective, normalized by 16 cards:
    // the winner collects (remaining x2-if-shutdown) of both losers; a loser
    // with all 16 cards never played pays double.
    double payoff(int seat) const;

    int prevSeat() const { return (turn + 2) % 3; }
    int nextSeat() const { return (turn + 1) % 3; }
};

}  // namespace pdk
