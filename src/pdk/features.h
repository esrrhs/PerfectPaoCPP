// Node / action feature engineering for Paodekuai, adapted from the
// PerfectDou feature scheme to a symmetric 3-player, suit-less game.
//
// Imperfect node features (observable by the acting player, relative seats):
//   card matrices of 11 x 13
//     own hand, both opponents combined, own/prev/next played cards,
//     own/prev/next last move, last 15 moves (history)
//   + one-hot hand counts of prev/next (16+16), one-hot bomb count (9),
//     control flag, reported-next / reported-prev flags
// Additional perfect features (critic only):
//   prev hand matrix + next hand matrix + 2 min-step scalars
//
// Action features: 11 x 13 matrix of the concrete move plus 6 dynamic
// scalars.
#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "pdk/cards.h"
#include "pdk/game.h"
#include "pdk/moves.h"

namespace pdk {

constexpr int kCardRows = 11;             // 4 count rows + 7 pattern rows
constexpr int kCardMat = kCardRows * kRanks;  // 143
constexpr int kHistoryLen = 15;
constexpr int kHistoryGroups = 5;         // 5 LSTM steps x 3 moves
constexpr int kStaticMatrices = 8;
constexpr int kImpMatrices = kStaticMatrices + kHistoryLen;  // 23
constexpr int kImpBin = kImpMatrices * kCardMat;             // 3289
constexpr int kHandHot = kHandSize;                          // 16
constexpr int kBombHot = 9;
constexpr int kImpScalars = 2 * kHandHot + kBombHot + 3;     // 44
constexpr int kImpSize = kImpBin + kImpScalars;  // stored transition feature
constexpr int kNodeBin = kStaticMatrices * kCardMat;  // 1144
constexpr int kNodeSize = kNodeBin + kImpScalars;     // 1188
constexpr int kExtraBin = 2 * kCardMat;               // 286
constexpr int kExtraScalars = 2;
constexpr int kExtraSize = kExtraBin + kExtraScalars;  // 288
constexpr int kActionExtra = 6;
constexpr int kActionSize = kCardMat + kActionExtra;   // 149

struct EncodedState {
    std::array<uint8_t, kImpBin> imp{};
    std::array<float, kImpScalars> scalar{};
    std::array<uint8_t, kExtraBin> extra{};
    std::array<float, kExtraScalars> extraScalar{};
};

struct LegalOption {
    int abstractId;  // local slot 0..n-1 within this decision point
    CardSet concrete;
    std::array<float, kActionSize> feature{};
};

// Encodes the node for the player whose turn it currently is.
EncodedState encodeState(const Game& g);

// One option per concrete legal play.
std::vector<LegalOption> legalOptions(const Game& g);
void legalOptions(const Game& g, std::vector<LegalOption>& out);

// Writes the generic 11x13 card matrix for any card set.
void cardMatrix(const CardSet& cs, uint8_t* out);

}  // namespace pdk
